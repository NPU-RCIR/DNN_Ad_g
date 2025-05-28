# data_processor.py
import numpy as np
import os

class SpaceNetDataProcessor:
    def __init__(self, data_root='data'):
        self.data_root = data_root
        self.create_dirs()

    def create_dirs(self):
        """创建数据存储目录结构"""
        dirs = ['train', 'test', 'validation']
        for d in dirs:
            path = os.path.join(self.data_root, d)
            os.makedirs(path, exist_ok=True)

    def process_single_trajectory(self, X_train_path, H_train_path, X_test_path, H_test_path, X_long_path):
        """
        处理单条轨迹数据
        :param X_path: 状态数据路径 (shape: 4, n_steps, 9)
        :param T_path: 拉力数据路径 (shape: 4, n_steps, 3)
        """
        # 加载原始数据
        train_features = np.load(X_train_path)  # (24,subtraj_length, 9)
        train_labels = np.load(H_train_path)  # (24, ubtraj_length)
        test_features = np.load(X_test_path)  # (6, subtraj_length, 9)
        test_labels = np.load(H_test_path)  # (6, subtraj_length)
        long_features = np.load(X_long_path)  # (6, n_step, 9)
        
        # 提取特征和标签
        
        # 保存数据
        n_steps = train_features.shape[1]
        self._save_dataset(
            train_features, train_labels,
            test_features, test_labels, long_features,
            n_steps=n_steps
        )

    def _build_dataset(self, X_train, H_train, X_test, H_test):
        """
        构建带邻居的特征矩阵和标签 0,1,2为训练集，3为测试集
        :param states: 状态数据 ( n_steps, 9)
        :param h: 拉力数据 ( n_steps)
        :return: (features, labels)
        """
        n_steps = X_train.shape[0]
        features = np.zeros((n_steps, 9))
        labels = np.zeros(n_steps)

        # return features[:3,:,:], labels[:3,:,:], features[3,:,:], labels[3,:,:]
        return features, labels, features, labels

    def _save_dataset(self, train_features, train_labels, test_features, test_labels, long_features, n_steps):
        """保存处理后的数据集"""
        n_traj_train = train_features.shape[0]
        n_traj_test = test_features.shape[0]

        # 训练集
        train_name = f"{n_traj_train}_{n_steps}_9"
        np.save(os.path.join(self.data_root, 'train', f'{train_name}_features.npy'), train_features)
        np.save(os.path.join(self.data_root, 'train', f'{train_name}_labels.npy'), train_labels)
        
        # 测试集
        test_name = f"{n_traj_test}_{n_steps}_9"  # 1个卫星，n_steps时间步，18维特征
        np.save(os.path.join(self.data_root, 'test', f'{test_name}_features.npy'), test_features)
        np.save(os.path.join(self.data_root, 'test', f'{test_name}_labels.npy'), test_labels)

        # 验证集
        np.save(os.path.join(self.data_root, 'validation', f'long_features.npy'), long_features)

    def load_dataset(self, dataset_type='train'):
        """
        加载处理后的数据集
        :param dataset_type: 'train' 或 'test'
        :return: (features, labels)
        """
        # 自动查找最新保存的文件
        feature_files = [f for f in os.listdir(os.path.join(self.data_root, dataset_type)) 
                        if f.endswith('features.npy')]
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {dataset_type} directory")
        
        # 加载最新文件
        feature_file = sorted(feature_files)[-1]
        label_file = feature_file.replace('features', 'labels')
        
        features = np.load(os.path.join(self.data_root, dataset_type, feature_file))
        labels = np.load(os.path.join(self.data_root, dataset_type, label_file))
        return features, labels

    # 加载验证时所需数据
    def load_validation_dataset(self, dataset_type='validation'):
        # 自动查找最新保存的文件
        all_files = os.listdir(os.path.join(self.data_root, dataset_type))
        feature_files = [f for f in all_files if f.endswith('features.npy')]
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {dataset_type} directory")

        # 构建三个特征文件和标签文件路径
        features_path = os.path.join(self.data_root, dataset_type, f"long_features.npy")

        features = np.load(features_path)


        return features

# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = SpaceNetDataProcessor(data_root='data')
    
    # 处理原始数据（假设已生成X.npy和T.npy）
    processor.process_single_trajectory(
        X_train_path='X_train.npy',
        H_train_path='H_train.npy',
        X_test_path='X_test.npy',
        H_test_path='H_test.npy',
        X_long_path='X_long.npy'
    )
    
    # 验证数据加载
    train_features, train_labels = processor.load_dataset('train')
    print(f"训练集形状: 特征 {train_features.shape}, 标签 {train_labels.shape}")
    
    test_features, test_labels = processor.load_dataset('test')
    print(f"测试集形状: 特征 {test_features.shape}, 标签 {test_labels.shape}")

    long_features = processor.load_validation_dataset('validation')
    print(f"验证集形状: 特征 {long_features.shape}")