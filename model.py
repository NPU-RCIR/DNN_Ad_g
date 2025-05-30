# model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN

class NetEstimator(nn.Module):
    def __init__(self, input_dim=9, output_dim=3, hidden_dim=256, delta_p_dim=12):
        super(NetEstimator, self).__init__()
        self.delta_p_dim = delta_p_dim
        self.output_dim = output_dim

        # 调整输入维度（原特征 + 前一时刻h）
        self.ad_input_dim = input_dim + output_dim
        self.g_input_dim = delta_p_dim + output_dim

        self.last_g_matrix = None  # 新增：存储最后生成的g矩阵
        # self.g_matrix = nn.Parameter(torch.randn(3, 12) * 0.01)  # 初始化为小随机值

        # f(x)网络：学习静态映射 h = f(x)
        # self.f_net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        # Ad网络：每个delta_p对应一个控制系数矩阵
        self.ad_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),  # 新增层归一化
                # nn.LeakyReLU(negative_slope=0.01),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),  # 新增层归一化
                # nn.LeakyReLU(negative_slope=0.01),
                nn.ReLU(),
                nn.Linear(hidden_dim, 9),
                # nn.Tanh()  # 新增Tanh
            )

        # g网络：每个delta_p对应一个控制系数矩阵
        self.g_net = nn.Sequential(
                nn.Linear(delta_p_dim, hidden_dim),  # 输入单个delta_p值
                # nn.LayerNorm(hidden_dim),  # 新增层归一化
                # nn.LeakyReLU(negative_slope=0.01),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                # nn.LayerNorm(hidden_dim),  # 新增层归一化
                # nn.LeakyReLU(negative_slope=0.01),
                nn.ReLU(),
                nn.Linear(hidden_dim, 36),
                # nn.Tanh()  # 新增Tanh
            )  # 四个卫星对应的g网络

    #     self._init_weights()  # 新增初始化方法
    #
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if m is self.ad_net[-1]:  # ad_net的输出层
    #                 nn.init.normal_(m.weight, mean=0, std=0.01)
    #             elif m is self.g_net[-1]:  # g_net的输出层
    #                 nn.init.normal_(m.weight, mean=0, std=0.001)
    #             else:
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x_t, u_t, h_prev):
        """
        输入:
            x: [batch, seq_len, input_dim]
            u: [batch, seq_len, 12]
        输出:
            h_pred: [batch, seq_len, 3]
        """
        batch_size, _ = x_t.size()

        # h_pred = torch.zeros(batch_size, seq_len, 3).to(x.device)

        # 初始化h_prev
        # if h_init is None:
        #     h_prev = torch.zeros(batch_size, 3).to(x.device)
        # else:
        #     h_prev = h_init
        # a = h_prev.shape

        # for t in range(seq_len):
        # 获取当前时间步特征和控制量
        # x_t = x[:, t, :]  # [batch, input_dim+delta_p_dim]
        # u_t = u[:, t, :]  # [batch, 12]
        # h_prev = h_init[:, t, :]

        # 分割输入
        delta_p = x_t[:, -self.delta_p_dim:]  # [batch, 12]
        other_features = x_t[:, :-self.delta_p_dim]  # [batch, 9]

        # 方法1: 平均池化（推荐）
        aggregated_delta_p = delta_p.mean(dim=0, keepdim=True)  # [1, 12]

        # x_h_t = torch.cat((other_features, h_prev), dim=1)
        # 计算基础h值
        # h_current = self.f_net(other_features)  # [batch, 3]
        ad = self.ad_net(other_features).view(-1, 3, 3)
        assert not torch.isnan(ad).any(), "NaN in ad matrix"

        # 生成12x12的G矩阵
        # g = self.g_net(delta_p).view(-1, 3, 12)  # [batch, 3, 12]
        # 生成共享g矩阵
        g_flat = self.g_net(aggregated_delta_p)  # [1, 36]
        g = g_flat.view(1, 3, 12)  # [1, 3, 12]

        # 扩展g矩阵以匹配批次大小
        g_expanded = g.expand(batch_size, -1, -1)  # [batch, 3, 12]
        self.last_g_matrix = g.detach().cpu()  # 新增：捕获并存储g矩阵

        # 计算G*u并求和
        u_total = u_t.unsqueeze(-1)  # [batch, 12, 1]
        sum_gu = torch.matmul(g_expanded, u_total).squeeze(-1)  # [batch, 12]
        # gu_split = gu_total.view(batch_size, 4, 3)  # 拆分为4个3维
        # sum_gu = gu_split.sum(dim=1)  # [batch, 3]

        # 应用离散化公式：h_{k+1} = Ad*h_k + (Σg*u)
        h_next = torch.matmul(ad, h_prev.unsqueeze(-1)).squeeze(-1) + sum_gu

        # h_pred[:, t, :] = h_next
        # h_prev = h_next.detach()  # 截断梯度传播

        return h_next


def get_loss_fn():
    return nn.MSELoss()