# model.py
import torch
import torch.nn as nn

class NetEstimator(nn.Module):
    def __init__(self, input_dim=9, output_dim=3, hidden_dim=256, delta_p_dim=12):
        super(NetEstimator, self).__init__()
        self.delta_p_dim = delta_p_dim
        self.output_dim = output_dim

        # 调整输入维度（原特征 + 前一时刻h）
        self.ad_input_dim = input_dim + output_dim
        self.g_input_dim = delta_p_dim + output_dim

        # f(x)网络：学习静态映射 h = f(x)
        # self.f_net = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)
        # )
        # LSTM 编码器：分别处理其他特征和 delta_p
        self.lstm_features = nn.LSTM(
            input_size=input_dim,  # 输入其他特征（非 delta_p）
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )
        self.lstm_delta_p = nn.LSTM(
            input_size=delta_p_dim,  # 输入 delta_p
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=2,
            dropout=0.2
        )

        # Ad网络：每个delta_p对应一个控制系数矩阵
        self.ad_net = nn.Sequential(
                nn.Linear(hidden_dim + output_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 新增层归一化
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(hidden_dim, 9),
                nn.Tanh()  # 新增Tanh
            )

        # g网络：每个delta_p对应一个控制系数矩阵
        self.g_net = nn.Sequential(
                nn.Linear(hidden_dim + output_dim, hidden_dim),  # 输入单个delta_p值
                nn.LayerNorm(hidden_dim),  # 新增层归一化
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(hidden_dim, 144),
                nn.Tanh()  # 新增Tanh
            )  # 四个卫星对应的g网络

    #     self._init_weights()  # 新增初始化方法
    #
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if m is self.ad_net[-2]:  # ad_net的输出层
    #                 nn.init.normal_(m.weight, mean=0, std=0.01)
    #             elif m is self.g_net[-2]:  # g_net的输出层
    #                 nn.init.normal_(m.weight, mean=0, std=0.001)
    #             else:
    #                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x_seq, u_seq, h_prev):
        """
        输入:
            x: [batch, seq_len, input_dim]
            u: [batch, seq_len, 12]
        输出:
            h_pred: [batch, seq_len, 3]
        """
        batch_size, seq_len, _ = x_seq.size()

        # 分割输入
        delta_p = x_seq[:, :, -self.delta_p_dim:]  # [batch, 12]
        other_features = x_seq[:, :, :-self.delta_p_dim]  # [batch, 9]

        # 编码其他特征的时间序列
        features_out, _ = self.lstm_features(other_features)  # [batch, seq_len, hidden_dim]
        # 编码 delta_p 的时间序列
        delta_p_out, _ = self.lstm_delta_p(delta_p)  # [batch, seq_len, hidden_dim]

        h_preds = []
        h_current = h_prev.clone()


        for t in range(seq_len):
            feature_t = features_out[:, t, :]  # [batch, hidden_dim]
            delta_p_t = delta_p_out[:, t, :]  # [batch, hidden_dim]

            ad_input = torch.cat([feature_t, h_current], dim=1)
            ad = self.ad_net(ad_input).view(-1, 3, 3)   # [batch, 3, 3]
            assert not torch.isnan(ad).any(), "NaN in ad matrix"

            # 生成12x12的G矩阵
            g_input = torch.cat([delta_p_t, h_current], dim=1)  # [batch, hidden_dim+output_dim]
            g = self.g_net(g_input).view(-1, 12, 12)  # [batch, 12, 12]
            assert not torch.isnan(g).any(), "NaN in g matrix"

            # 计算G*u并求和
            u_total = u_seq[:, t, :].unsqueeze(-1)  # [batch, 12, 1]
            gu_total = torch.matmul(g, u_total).squeeze(-1)  # [batch, 12]
            gu_split = gu_total.view(batch_size, 4, 3)  # 拆分为4个3维
            sum_gu = gu_split.sum(dim=1)  # [batch, 3]

            # 应用离散化公式：h_{k+1} = Ad*h_k + Bd*(Σg*u)
            h_next = torch.matmul(ad, h_current.unsqueeze(-1)).squeeze(-1) + sum_gu
            h_preds.append(h_next.unsqueeze(1))
            h_current = h_next.detach()  # 截断梯度

        h_pred = torch.cat(h_preds, dim=1)  # [batch, seq_len, 3]

        return h_pred


def get_loss_fn():
    return nn.MSELoss()