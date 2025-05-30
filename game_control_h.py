import mujoco
import mujoco_viewer
import numpy as np
import copy
import time
import os

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import ADP, error_dynamic, game_dynamic, rotation_matrix_from_quaternion

# 仿真时间步长
num_trajectories = 10  # 长轨迹数量
duration = 30
dt = 0.001
n_steps = int(duration / dt)  # 0.001
subtraj_length = 1000

# 最大控制力和力矩
u_max = 6
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
    a_formatted = "_".join([f"{x:.2f}" for x in a])
    v_formatted = "_".join([f"{x:.2f}" for x in v])

    # 定义子文件夹路径
    subfolder = os.path.join(folder, f"s_{s} a_{a_formatted} v_{v_formatted}")

    # 如果子文件夹不存在，则创建
    os.makedirs(subfolder, exist_ok=True)

    # 保存数据到 .npy 文件
    np.save(os.path.join(subfolder, "E.npy"), Error)
    np.save(os.path.join(subfolder, "C.npy"), Weights)
    np.save(os.path.join(subfolder, "W.npy"), Configuration)


# 加载模型
model = mujoco.MjModel.from_xml_path("E:/DNN_pva_copy/TSNR/net_11_11_h.mjcf")
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

critic_h = ADP(dt)

control_0 = np.zeros(6)
control_1 = np.zeros(6)
control_2 = np.zeros(6)
control_3 = np.zeros(6)

control_h_0 = np.zeros(3)
control_h_1 = np.zeros(3)
control_h_2 = np.zeros(3)
control_h_3 = np.zeros(3)

V_0 = 0
V_1 = 0
V_2 = 0
V_3 = 0

# 迭代参数
ksi = 0.001
V_max = 1000

V_h_max = 1000

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
v_c_0 = 0.5
a_c_d = 0.5
traj = generate_trajectory(v_c_0, a_c_d, n_steps, dt)

mujoco.mj_step(model, data)
data.qvel[0:3] = np.array([0.5, 0, 0])
data.qvel[6:9] = np.array([0.5, 0, 0])
data.qvel[12:15] = np.array([0.5, 0, 0])
data.qvel[18:21] = np.array([0.5, 0, 0])
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

    h_d = np.array([1.1, 0.01, 0.01])
    h_e = h - h_d
    d_h_e = ( - h)/dt

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

    if V_h_max > ksi:
        W_h, V_h_new = critic_h.update_weights_h(h_e, d_h_e, control_0, control_1, control_2, control_3)

        V_h_max = V_h_new

        V_h = V_h_new

    # 计算控制量 control = u, tau
    control_0 = critic_0.NE(g_x_0)
    control_1 = critic_0.NE(g_x_1)
    control_2 = critic_0.NE(g_x_2)
    control_3 = critic_0.NE(g_x_3)

    # 传入g0-3
    checkpoint = torch.load("models/best_model.pth", weights_only=False)
    g_matrix = checkpoint['g_matrix']  # 获取保存的g矩阵
    g_all = g_matrix.squeeze(0)
    g_0 = g_all[:, :3]
    g_1 = g_all[:, 3:6]
    g_2 = g_all[:, 6:9]
    g_3 = g_all[:, 9:12]

    control_h_0, control_h_1, control_h_2, control_h_3, = critic_h.NE_h(h_e , g_0, g_1, g_2, g_3)

    u = np.zeros(24)
    u[0:3] = np.clip(control_0[:3] + control_h_0, -u_max, u_max)
    u[3:6] = np.clip(control_0[3:], -t_max, t_max)
    u[6:9] = np.clip(control_1[:3] + control_h_1, -u_max, u_max)
    u[9:12] = np.clip(control_1[3:], -t_max, t_max)
    u[12:15] = np.clip(control_2[:3] + control_h_2, -u_max, u_max)
    u[15:18] = np.clip(control_2[3:], -t_max, t_max)
    u[18:21] = np.clip(control_3[:3] + control_h_3, -u_max, u_max)
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

    # 渲染
    # viewer.render()
    # if not viewer.is_alive:
    #     break

# viewer.close()

# 存储仿真结果
save_data(Error, Weights, Configuration, s=25, a=a_c_d[0], v=v_c_0[0], folder="data_EWC")
