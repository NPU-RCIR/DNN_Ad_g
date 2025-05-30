# train.py
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from data_processor import SpaceNetDataProcessor
from model import NetEstimator, get_loss_fn
import os


def main():
    # 配置参数
    BATCH_SIZE = 64
    EPOCHS = 200
    HIDDEN_DIM = 512
    LEARNING_RATE = 0.001
    SAVE_DIR = "models"

    predict_time = 20

    # 初始化组件
    processor = SpaceNetDataProcessor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_features, train_labels = processor.load_dataset('train')
    test_features, test_labels = processor.load_dataset('test')
    validation_features = processor.load_validation_dataset('validation')
    # 模型的训练目标就是学习特征和标签之间的映射关系

    # 数据预处理：标准化
    # 数据预处理：标准化
    train_mean = train_features[..., :-12].mean(axis=(0,1))
    train_std = train_features[..., :-12].std(axis=(0,1)) + 1e-8
    # validation_mean = validation_features[..., :-12].mean(axis=(0,1))
    # validation_std = validation_features[..., :-12].std(axis=(0,1)) + 1e-8
    train_features[..., :-12] = (train_features[..., :-12] - train_mean) / train_std
    test_features[..., :-12] = (test_features[..., :-12] - train_mean) / train_std

    u_mean = train_features[..., -12:].mean(axis=(0, 1))
    u_std = train_features[..., -12:].std(axis=(0, 1)) + 1e-8
    # validation_u_mean = validation_features[..., -12:].mean(axis=(0, 1))
    # validation_u_std = validation_features[..., -12:].std(axis=(0, 1)) + 1e-8
    train_features[..., -12:] = (train_features[..., -12:] - u_mean) / u_std
    test_features[..., -12:] = (test_features[..., -12:] - u_mean) / u_std
    # 训练数据的特征就被转换为均值为 0、标准差为 1 的分布。

    # 转换为PyTorch Tensor
    X_train = torch.FloatTensor(train_features).to(device)  # 9+3*4+4*3=33
    y_train = torch.FloatTensor(train_labels).to(device)
    X_test = torch.FloatTensor(test_features).to(device)
    y_test = torch.FloatTensor(test_labels).to(device)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 初始化模型和优化器
    model = NetEstimator(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = get_loss_fn()

    # 训练循环
    best_test_loss = float('inf')  # 将最佳测试损失初始化为正无穷大
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()  # 将模型设置为训练模式
        train_loss = 0.0

        for X_batch, y_batch in train_loader:  # num_batches = len(train_dataset) // BATCH_SIZE if !=0, num+1
            # X_batch: [batch, seq_len, 33]
            # y_batch: [batch, seq_len, 3]
            # a = X_batch.shape
            # b = y_batch.shape
            optimizer.zero_grad()

            # 分割输入和控制量
            u_batch = X_batch[:, :, -12:]  # 控制输入 [batch, seq_len, 12]
            features = X_batch[:, :, :-12]  # [other_features (9) | delta_p (12)]

            y = y_batch[:, predict_time:, :]

            for step in range(predict_time - 1):
                end = features.shape[1] - (predict_time - step)

                u = u_batch[:, step:end, :]
                x = features[:, step:end, :]
                
                batch_size, seq_len, _ = x.shape
                
                # h_prev = y_batch[:, t, :].detach()  # 使用真值作为下一步输入
                # 前向传播
                if step == 0:
                    h_init = y_batch[:, step:end, :]
                    h_preds = []
                    for t in range(seq_len):
                        x_t = x[:, t, :]  # [batch, 9]
                        u_t = u[:, t, :]  # [batch, 12]
                        h_prev = h_init[:, t, :]  # [batch, 3]

                        h_next = model(x_t, u_t, h_prev)
                        h_preds.append(h_next.unsqueeze(1))
                    h_pred = torch.cat(h_preds, dim=1)  # [batch, seq_len, 3]
                else:
                    # 后续step用上一个step的h_pred作为h_init
                    h_init = h_pred.detach()  # 不反传梯度
                    h_preds = []
                    for t in range(seq_len):
                        x_t = x[:, t, :]
                        u_t = u[:, t, :]
                        h_prev = h_init[:, t, :]
                        h_next = model(x_t, u_t, h_prev)
                        h_preds.append(h_next.unsqueeze(1))
                    h_pred = torch.cat(h_preds, dim=1)

                # a = h_pred.shape
            loss = loss_fn(h_pred, y)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)

        # 验证
        model.eval()
        test_loss = 0.0
        with torch.no_grad():  # 上下文管理器，用于关闭梯度计算。在验证阶段，不需要计算梯度，关闭梯度计算可以节省内存并提高计算速度。
            for X_batch, y_batch in test_loader:
                a = X_batch.shape
                b = y_batch.shape
                u_batch = X_batch[:, :, -12:]  # 控制输入 [batch, 12]
                features = X_batch[:, :, :-12]  # [other_features (9) | delta_p (12)]
                # h_init = y_batch[:, 0, :]
                y = y_batch[:, predict_time:, :]

                # for step in range(predict_time - 1):
                step = 0
                end = features.shape[1] - (predict_time - step)
                u = u_batch[:, step:end, :]
                x = features[:, step:end, :]

                batch_size, seq_len, _ = x.shape
                for step in range(predict_time - 1):
                    end = features.shape[1] - (predict_time - step)
                    u = u_batch[:, step:end, :]
                    x = features[:, step:end, :]

                    if step == 0:
                        h_init = y_batch[:, step:end, :]
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
                        h_pred = torch.cat(h_preds, dim=1)
                # pred_h_next = model(x, u, h_init)
                test_loss += loss_fn(h_pred, y).item() * X_batch.size(0)

        # 计算平均损失
        train_loss = train_loss / len(train_dataset)
        test_loss = test_loss / len(test_dataset)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss

            # 获取模型生成的g矩阵
            g_matrix = model.last_g_matrix.numpy() if model.last_g_matrix is not None else None

        # train.py (片段)
        torch.save({
            'model': model.state_dict(),
            'mean': train_mean,
            'std': train_std,
            'u_mean': u_mean,
            'u_std': u_std,
            # 'mean_long': validation_mean,
            # 'std_long': validation_std,
            'hidden_dim': HIDDEN_DIM,  # 新增此行
            'g_matrix': g_matrix  # 新增：保存g矩阵
        }, f"{SAVE_DIR}/best_model.pth")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    main()