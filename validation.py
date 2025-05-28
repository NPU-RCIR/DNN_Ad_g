# validate.py
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_processor import SpaceNetDataProcessor
from model import NetEstimator
import matplotlib.pyplot as plt

def main():
    # 加载配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SpaceNetDataProcessor()
    
    # 加载测试数据
    test_features1 = processor.load_validation_dataset('validation')
    test_labels = np.load('H_long.npy')
    # 为匹配model，增加一维
    test_features1 = np.expand_dims(test_features1, axis=0)
    test_labels = np.expand_dims(test_labels, axis=0)

    # 加载模型和超参数
    checkpoint = torch.load("models/best_model.pth", weights_only=False)
    model = NetEstimator(input_dim=9, output_dim=3, hidden_dim=checkpoint['hidden_dim'], delta_p_dim=12).to(device)  # 动态设置维度
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # def normalize(data):
    #     """处理[num_traj, n_steps, features]形状数据"""
    #     mean = data.mean(axis=(0,1))
    #     std = data.std(axis=(0,1)) + 1e-8
    #     return (data - mean) / std, mean, std

    # 预处理测试数据
    test_features1[..., :-12] = (test_features1[..., :-12] - checkpoint['mean']) / checkpoint['std']
    test_features1[..., -12:] = (test_features1[..., -12:] - checkpoint['u_mean']) / checkpoint['u_std']
    original_shape = test_features1.shape

    # 展平为 (num_traj * n_steps, 12)
    # test_features1 = test_features1.reshape(-1, original_shape[-1])

    X_test1 = torch.FloatTensor(test_features1).to(device)  # [1, n_steps, 33]
    y_test1 = torch.FloatTensor(test_labels).to(device)

    # test_dataset = TensorDataset(X_test1, X_test2, X_test3)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # # 显存不足，分批预测
    # preds = []
    # model.eval()
    # with torch.no_grad():
    #     for batch in test_loader:
    #         batch_X1, batch_X2, batch_X3 = [t.to(device) for t in batch]
    #         batch_pred = model(batch_X1, batch_X2, batch_X3).cpu().numpy()
    #         preds.append(batch_pred)
    #         del batch_X1, batch_X2, batch_X3, batch_pred
    #         torch.cuda.empty_cache()
    # 预测并恢复形状

    u_ = X_test1[:, :, -12:]  # 控制输入 [batch, 12]
    features = X_test1[:, :, :-12]  # [other_features (9) | delta_p (12)]

    # u = u_[:, :-1, :]
    # x = features[:, :-1, :]
    # # h_init = y_test1[:, :-1, :]
    # h_init = y_test1[:, 0, :]
    # batch_size, seq_len, _ = x.shape
    # h_preds = []
    with torch.no_grad():
        
        # for t in range(seq_len):
        #     x_t = x[:, t, :]  # [batch, 9]
        #     u_t = u[:, t, :]  # [batch, 12]
        #     # h_prev = h_init[:, t, :]
        #     if t == 0:
        #         h_next = model(x_t, u_t, h_init)
        #         h_prev = h_next.detach()
        #     # h_prev = h_init[:, t, :]  # [batch, 3]
        #     else:
        #         h_next = model(x_t, u_t, h_prev)
        #         h_prev = h_next.detach()
        #
        #     # h_next = model(x_t, u_t, h_prev)
        #
        #     h_preds.append(h_next.unsqueeze(1))
        predict_time = 20
        for step in range(predict_time - 1):
            end = features.shape[1] - (predict_time - step)
            u = u_[:, step:end, :]
            x = features[:, step:end, :]

            if step == 0:
                h_init = y_test1[:, step:end, :]
                h_preds = []
                for t in range(x.shape[1]):
                    x_t = x[:, t, :]
                    u_t = u[:, t, :]
                    h_prev = h_init[:, t, :]
                    h_next = model(x_t, u_t, h_prev)
                    h_preds.append(h_next.unsqueeze(1))
                h_pred = torch.cat(h_preds, dim=1)
            else:
                h_init = h_pred
                h_preds = []
                for t in range(x.shape[1]):
                    x_t = x[:, t, :]
                    u_t = u[:, t, :]
                    h_prev = h_init[:, t, :]
                    h_next = model(x_t, u_t, h_prev)
                    h_preds.append(h_next.unsqueeze(1))

        # 前向传播
        h_pred = torch.cat(h_preds, dim=1).cpu().numpy()  # [batch, seq_len, 3]

        # preds = model(x, u, h_init).cpu().numpy()
    # 合并结果并恢复形状
    # preds = np.concatenate(preds, axis=0)
    # preds = preds.reshape(original_shape[0], original_shape[1], -1)  # (num_traj, n_steps, 3)
    # test_labels = test_labels.reshape(original_shape[0], original_shape[1], -1)


    # 恢复为原始轨迹形状 [num_traj, n_steps, 3]
    # preds = preds.reshape(original_shape[0], original_shape[1], -1)
    # test_labels = test_labels.reshape(original_shape[0], original_shape[1], -1)

    # 可视化结果
    # plt.figure(figsize=(12, 8))
    # for sat in range(preds.shape[0]):
    #     for dim in range(3):
    #         plt.subplot(4, 3, sat*3 + dim + 1)
    #         plt.plot(test_labels[sat, :, dim], label='True')
    #         plt.plot(preds[sat, :, dim], label='Predicted')
    #         plt.title(f"Sat {sat} F{dim+1}")
    #         plt.legend()
    # plt.tight_layout()
    # plt.savefig("validation_results.png")
    # plt.show()
    # 绘制网兜深度h，xyz方向上的预测与真实值
    plt.figure(figsize=(12, 8))
    for dim in range(3):
        plt.subplot(3, 1, dim+1)
        plt.plot(test_labels[0, 20:, dim], label='True')
        plt.plot(h_pred[0, :, dim], label='Predicted')
        plt.legend()
    plt.tight_layout()
    plt.savefig("validation_results.png")
    plt.show()

if __name__ == "__main__":
    main()