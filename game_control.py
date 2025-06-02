import mujoco
import mujoco_viewer
import numpy as np
import copy
import time
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import ADP, error_dynamic, game_dynamic, rotation_matrix_from_quaternion

# 仿真时间步长
num_trajectories = 10  # 长轨迹数量
duration = 30
dt = 0.001
n_steps = int(duration / dt)  # 0.001
subtraj_length = 1000
total_subtraj = (n_steps * num_trajectories) // subtraj_length
train_ratio = 0.8
num_train = int(total_subtraj * train_ratio)
num_test = total_subtraj - num_train

# 最大控制力和力矩
u_max = 5
t_max = 0.5

# 图论
A_c = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])
B_c = np.diag([1, 1, 1, 1])
a_0 = A_c[0, 1] + A_c[0, 2] + B_c[0, 0]
a_01 = A_c[0, 1]
a_02 = A_c[0, 2]
a_1 = A_c[1, 0] + A_c[1, 3] + B_c[1, 1]
a_10 = A_c[1, 0]
a_13 = A_c[1, 3]
a_2 = A_c[2, 0] + A_c[2, 3] + B_c[2, 2]
a_20 = A_c[2, 0]
a_23 = A_c[2, 3]
a_3 = A_c[3, 1] + A_c[3, 2] + B_c[3, 3]
a_31 = A_c[3, 1]
a_32 = A_c[3, 2]
# 期望轨迹
# current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在的文件夹路径
# path_planning_dir = os.path.join(current_dir, "Path Planning")  # 构造 path_planning 文件夹路径
# npy_file_path = os.path.join(path_planning_dir, "straight_traj.npy")  # 构造 .npy 文件的完整路径
# traj = np.load(npy_file_path)  # 读取 .npy 文件
# traj = np.load('traj.npy')
np.random.seed(42)
v_c_0_list = np.zeros((num_trajectories, 3))
a_c_d_list = np.zeros((num_trajectories, 3))
v_c_0_list[:, 0] = np.random.uniform(0.4, 0.5, num_trajectories)
a_c_d_list[:, 0] = np.random.uniform(0.3, 0.4, num_trajectories)
# 生成带姿态直线轨迹
def generate_trajectory(v_c_0, a_c_d, n_steps, dt):
    """生成包含姿态的期望轨迹"""
    trajectory = np.zeros((4, n_steps, 19))
    delta_p_list = [
        np.array([0, 2.75, 2.75]),
        np.array([0, -2.75, 2.75]),
        np.array([0, 2.75, -2.75]),
        np.array([0, -2.75, -2.75])
    ]
    for i in range(n_steps):
        t = i * dt
        p_c = np.array([-10, 0, 0]) + v_c_0 * t + 0.5 * a_c_d * t**2
        v_c = v_c_0 + a_c_d * t
        for sat_idx in range(4):
            delta_p = delta_p_list[sat_idx]
            trajectory[sat_idx, i, 0:3] = p_c + delta_p
            trajectory[sat_idx, i, 3:6] = v_c
            trajectory[sat_idx, i, 6:9] = a_c_d
            trajectory[sat_idx, i, 9:13] = np.array([1.0, 0.0, 0.0, 0.0])  # 固定四元数
            trajectory[sat_idx, i, 13:16] = np.zeros(3)  # 角速度
            trajectory[sat_idx, i, 16:19] = np.zeros(3)  # 角加速度

    return trajectory

def save_data(Error, Weights, Configuration, s, a, v, folder="data"):
    # 格式化a和v为两位小数字符串
    a_formatted = f"{a:.2f}"
    v_formatted = f"{v:.2f}"

    # 定义子文件夹路径
    subfolder = os.path.join(folder, f"s_{s} a_{a_formatted} v_{v_formatted}")

    # 如果子文件夹不存在，则创建
    os.makedirs(subfolder, exist_ok=True)

    # 保存数据到 .npy 文件
    np.save(os.path.join(subfolder, "E.npy"), Error)
    np.save(os.path.join(subfolder, "W.npy"), Weights)
    np.save(os.path.join(subfolder, "C.npy"), Configuration)

def run_simulation(v_c_0, a_c_d):
    # 加载模型
    model = mujoco.MjModel.from_xml_path("TSNR/net_11_11_h.mjcf")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # 渲染
    # viewer = mujoco_viewer.MujocoViewer(model, data)

    # 初始化
    Error = np.zeros((4, n_steps, 16))
    Weights = np.zeros((4, n_steps, 18))
    Configuration = np.zeros((n_steps, 9))

    critic_0 = ADP(dt)
    critic_1 = ADP(dt)
    critic_2 = ADP(dt)
    critic_3 = ADP(dt)

    control_0 = np.zeros(6)
    control_1 = np.zeros(6)
    control_2 = np.zeros(6)
    control_3 = np.zeros(6)

    V_0 = 0
    V_1 = 0
    V_2 = 0
    V_3 = 0

    # 迭代参数
    ksi = 0.001
    V_max = 1000

    # 初始化数据存储
    U_1 = np.zeros((n_steps,3))
    U_2 = np.zeros((n_steps,3))
    U_3 = np.zeros((n_steps,3))
    U_4 = np.zeros((n_steps,3))
    X_long = np.zeros((n_steps, 33))
    H_long = np.zeros((n_steps, 3))
    control_inputs = np.zeros((n_steps, 12))
    prev_v_c = np.zeros(3)

    # 生成期望轨迹
    traj = generate_trajectory(v_c_0, a_c_d, n_steps, dt)

    mujoco.mj_step(model, data)
    data.qvel[0:3] = np.array([v_c_0[0], 0, 0])  # 使用v_c_0的x分量
    data.qvel[6:9] = np.array([v_c_0[0], 0, 0])
    data.qvel[12:15] = np.array([v_c_0[0], 0, 0])
    data.qvel[18:21] = np.array([v_c_0[0], 0, 0])

    data.qacc[0:3] = np.array([a_c_d[0], 0, 0])  # 使用a_c_d的x分量
    data.qacc[6:9] = np.array([a_c_d[0], 0, 0])
    data.qacc[12:15] = np.array([a_c_d[0], 0, 0])
    data.qacc[18:21] = np.array([a_c_d[0], 0, 0])
    # data.qvel[-1] = 0.125
    # 期望速度 加速度
    # v_c_0 = np.array([0.5,0,0])
    # a_c_d = np.array([0.4,0,0])

    # mujoco.mj_step(model, data)
    for i in tqdm(range(n_steps), desc="仿真进度", ncols=100):  # 进度条

        # 期望轨迹 pd vd ud qd wd taud
        x_0_d = traj[0, i, :]
        x_1_d = traj[1, i, :]
        x_2_d = traj[2, i, :]
        x_3_d = traj[3, i, :]

        # 当前状态p v q w
        p_0 = data.body('MU_0').xpos.copy()
        v_0 = data.sensor('vel_0').data.copy()
        p_1 = data.body('MU_1').xpos.copy()
        v_1 = data.sensor('vel_1').data.copy()
        p_2 = data.body('MU_2').xpos.copy()
        v_2 = data.sensor('vel_2').data.copy()
        p_3 = data.body('MU_3').xpos.copy()
        v_3 = data.sensor('vel_3').data.copy()

        q_0 = data.body('MU_0').xquat.copy()
        w_0 = data.sensor('angvel_0').data.copy()
        q_1 = data.body('MU_1').xquat.copy()
        w_1 = data.sensor('angvel_1').data.copy()
        q_2 = data.body('MU_2').xquat.copy()
        w_2 = data.sensor('angvel_2').data.copy()
        q_3 = data.body('MU_3').xquat.copy()
        w_3 = data.sensor('angvel_3').data.copy()

        # 网口位置
        p_c = 0.25 * (p_0 + p_1 + p_2 + p_3)
        p_n = data.body('node_5_5').xpos.copy()
        v_c = 0.25 * (v_0 + v_1 + v_2 + v_3)

        # 计算加速度（首帧设为0）
        if i == 0:
            a_c = np.zeros(3)
        else:
            a_c = (v_c - prev_v_c) / dt

        # 更新前一时刻速度
        prev_v_c = v_c.copy()

        # 网兜深度
        h = p_c - p_n
        # h = rotation_matrix_from_quaternion(q_0) @ h
        # 坐标变换
        if h[0] > 0:
            h_norm = np.linalg.norm(h)
        else:
            h_norm = -np.linalg.norm(h)

        # 网口面积
        s = (np.linalg.norm(p_0 - p_1) - 0.5) ** 2
        if s > 25:
            s = 25

        # 状态
        x_0 = np.concatenate((p_0, v_0, q_0, w_0))
        x_1 = np.concatenate((p_1, v_1, q_1, w_1))
        x_2 = np.concatenate((p_2, v_2, q_2, w_2))
        x_3 = np.concatenate((p_3, v_3, q_3, w_3))

        # 状态误差和状态误差微分
        x_0_e, d_x_0_e = error_dynamic(x_0, x_0_d, control_0)
        x_1_e, d_x_1_e = error_dynamic(x_1, x_1_d, control_1)
        x_2_e, d_x_2_e = error_dynamic(x_2, x_2_d, control_2)
        x_3_e, d_x_3_e = error_dynamic(x_3, x_3_d, control_3)

        # 博弈状态和状态微分，协同一致性误差
        g_x_0, d_g_x_0 = game_dynamic(x_0_e, x_1_e, x_2_e, d_x_0_e, d_x_1_e, d_x_2_e, a_0, a_01, a_02)
        g_x_1, d_g_x_1 = game_dynamic(x_1_e, x_0_e, x_3_e, d_x_1_e, d_x_0_e, d_x_3_e, a_1, a_10, a_13)
        g_x_2, d_g_x_2 = game_dynamic(x_2_e, x_0_e, x_3_e, d_x_1_e, d_x_0_e, d_x_3_e, a_2, a_20, a_23)
        g_x_3, d_g_x_3 = game_dynamic(x_3_e, x_1_e, x_2_e, d_x_0_e, d_x_1_e, d_x_2_e, a_3, a_31, a_32)

        # 自适应动态规划
        if V_max > ksi:
            # 更新权重
            W_0, V_0_new = critic_0.update_weights(g_x_0, d_g_x_0, control_0, control_1, control_2)
            W_1, V_1_new = critic_1.update_weights(g_x_1, d_g_x_1, control_1, control_0, control_3)
            W_2, V_2_new = critic_2.update_weights(g_x_2, d_g_x_2, control_2, control_0, control_3)
            W_3, V_3_new = critic_3.update_weights(g_x_3, d_g_x_3, control_3, control_1, control_2)

            V_max = max(np.abs(V_0_new - V_0), np.abs(V_1_new - V_1), np.abs(V_2_new - V_2), np.abs(V_3_new - V_3))

            V_0 = V_0_new
            V_1 = V_1_new
            V_2 = V_2_new
            V_3 = V_3_new

        # 计算控制量 control = u, tau
        control_0 = critic_0.NE(g_x_0)
        control_1 = critic_0.NE(g_x_1)
        control_2 = critic_0.NE(g_x_2)
        control_3 = critic_0.NE(g_x_3)

        u = np.zeros(24)
        u[0:3] = np.clip(control_0[:3], -u_max, u_max)
        u[3:6] = np.clip(control_0[3:], -t_max, t_max)
        u[6:9] = np.clip(control_1[:3], -u_max, u_max)
        u[9:12] = np.clip(control_1[3:], -t_max, t_max)
        u[12:15] = np.clip(control_2[:3], -u_max, u_max)
        u[15:18] = np.clip(control_2[3:], -t_max, t_max)
        u[18:21] = np.clip(control_3[:3], -u_max, u_max)
        u[21:24] = np.clip(control_3[3:], -t_max, t_max)

        data.ctrl = u

        mujoco.mj_step(model, data)

        U_1[i, :] = u[0:3]
        U_2[i, :] = u[6:9]
        U_3[i, :] = u[12:15]
        U_4[i, :] = u[18:21]

        Error[0, i, :] = np.concatenate((x_0_e,u[0:3]))
        Error[1, i, :] = np.concatenate((x_0_e,u[6:9]))
        Error[2, i, :] = np.concatenate((x_0_e,u[12:15]))
        Error[3, i, :] = np.concatenate((x_0_e,u[18:21]))
        Weights[0, i, :] = W_0
        Weights[1, i, :] = W_1
        Weights[2, i, :] = W_2
        Weights[3, i, :] = W_3
        Configuration[i, 0] = s
        Configuration[i, 1] = h_norm
        Configuration[i, 2:5] = p_c
        Configuration[i, 5:9] = q_0


        X_long[i, 0:3] = p_c
        X_long[i, 3:6] = v_c
        X_long[i, 6:9] = a_c
        # 新 delta_p
        X_long[i, 9:12] = np.array([0, 2.75, 2.75])
        X_long[i, 12:15] = np.array([0, -2.75, 2.75])
        X_long[i, 15:18] = np.array([0, 2.75, -2.75])
        X_long[i, 18:21] = np.array([0, -2.75, -2.75])
        # u
        X_long[i, 21:24] = u[0:3]
        X_long[i, 24:27] = u[6:9]
        X_long[i, 27:30] = u[12:15]
        X_long[i, 30:33] = u[18:21]

        H_long[i, :] = h

        # 渲染
        # viewer.render()
        # if not viewer.is_alive:
        #     break

    # viewer.close()

    # 存储仿真结果
    save_data(Error, Weights, Configuration, s=25, a=a_c_d[0], v=v_c_0[0], folder="data_EWC")

    return X_long, H_long

def split_trajectory(X_long, H_long, subtraj_length):
    """将长轨迹分割为多个子轨迹"""
    total_steps = X_long.shape[0]  # 获取时间步总数 (30000)
    total_subtraj = total_steps // subtraj_length  # 应等于30
    split_X = []
    split_H = []

    for i in range(total_subtraj):
        # 提取每个子轨迹的片段
        X_sub = X_long[i * subtraj_length: (i + 1) * subtraj_length, :]
        H_sub = H_long[i * subtraj_length: (i + 1) * subtraj_length, :]
        # 调整维度为 (4, subtraj_length, 10) -> (subtraj_length, 4, 10) 以便于后续处理
        # X_sub = np.transpose(X_sub, (1, 0, 2))  # 维度变为 (1000, 4, 10)
        split_X.append(X_sub)
        split_H.append(H_sub)

    return np.array(split_X), np.array(split_H)

# 记录开始时间
start_time = time.time()

X_total = np.zeros((n_steps * num_trajectories, 33))
H_total = np.zeros((n_steps * num_trajectories, 3))
# 存储训练集和测试集
X_train, H_train = [], []
X_test, H_test = [], []

# 把num_trajectories个长轨迹按时间步堆叠
for traj_idx in range(num_trajectories):
    v_c_0 = v_c_0_list[traj_idx]
    a_c_d = a_c_d_list[traj_idx]
    print(f"Processing trajectory {traj_idx + 1}/{num_trajectories}...")
    X_long, H_long = run_simulation(v_c_0, a_c_d)
    X_total[traj_idx * n_steps : (traj_idx+1) * n_steps] = X_long
    H_total[traj_idx * n_steps : (traj_idx+1) * n_steps] = H_long

# 分割轨迹并调整维度
X_subs, H_subs = split_trajectory(X_total, H_total, subtraj_length)  # [sub_traj, sub_steps, x_dim]
# 划分训练集和测试集
X_train = X_subs[:num_train]
H_train = H_subs[:num_train]
X_test = X_subs[num_train:]
H_test = H_subs[num_train:]


# # 转换为NumPy数组并保存
# X_train = np.array(X_train)  # 形状 (24, 4, 1000, 10)
# H_train = np.array(H_train)  # 形状 (24, 1000, 3)
# X_test = np.array(X_test)    # 形状 (6, 4, 1000, 10)
# H_test = np.array(H_test)    # 形状 (6, 1000, 3)

# np.save('X_long.npy', X_long)
# np.save('H_long.npy', H_long)
# np.save('X_train.npy', X_train)
# np.save('H_train.npy', H_train)
# np.save('X_test.npy', X_test)
# np.save('H_test.npy', H_test)

# 绘制h的xyz变化
plt.figure(figsize=(12, 8))
for dim in range(3):
    plt.subplot(3,1,dim+1)
    plt.plot(H_long[:, dim], label='True')
    plt.legend()
plt.tight_layout()
plt.show()


# 绘制参数设置
# plt.figure(figsize=(12, 8))
# time_steps = np.arange(n_steps) * dt  # 时间轴（单位：秒）
#
# # 定义子图布局（4个MU，每个MU的3维输入分开展示）
# for mu_idx, (mu_control, mu_name) in enumerate(zip(
#         [U_1, U_2, U_3, U_4],
#         ["MU0", "MU1", "MU2", "MU3"]
# )):
#     plt.subplot(4, 3, mu_idx * 3 + 1)
#     plt.plot(time_steps, mu_control[:, 0], label='X轴')
#     plt.title(f"{mu_name} 控制输入（X轴）")
#     plt.xlabel('时间 (s)')
#     plt.ylabel('输入值')
#
#     plt.subplot(4, 3, mu_idx * 3 + 2)
#     plt.plot(time_steps, mu_control[:, 1], label='Y轴', color='orange')
#     plt.title(f"{mu_name} 控制输入（Y轴）")
#     plt.xlabel('时间 (s)')
#     plt.ylabel('输入值')
#
#     plt.subplot(4, 3, mu_idx * 3 + 3)
#     plt.plot(time_steps, mu_control[:, 2], label='Z轴', color='green')
#     plt.title(f"{mu_name} 控制输入（Z轴）")
#     plt.xlabel('时间 (s)')
#     plt.ylabel('输入值')
#
# plt.tight_layout()
# plt.show()

print(f"数据保存完成，训练集：{len(X_train)}条，测试集：{len(X_test)}条")
# 验证维度
print(f"X_train shape: {X_train.shape},X_test shape: {X_test.shape}")  # 应输出 (24, 1000, 4, 10)
print(f"H_train shape: {H_train.shape},H_test shape: {H_test.shape}")  # 应输出 (24, 1000, 3)

# 计算并显示运行时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"程序总运行时间: {elapsed_time:.2f}秒")