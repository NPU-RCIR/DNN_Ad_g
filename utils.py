import numpy as np

def quaternion_conjugate(q):
    # 计算四元数的共轭
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    # 计算两个四元数的乘积
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def skew_symmetric(v):
    # 计算矢量的叉乘矩阵（skew-symmetric matrix）
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])


def rotation_matrix_from_quaternion(q):
    # 从四元数计算旋转矩阵
    # 全局坐标系到空间飞网机器人系
    w, x, y, z = q
    sigma_ei0 = w
    sigma_eiv = np.array([x, y, z])

    I = np.eye(3)
    sigma_eiv_cross = skew_symmetric(sigma_eiv)

    return (sigma_ei0 ** 2 - np.dot(sigma_eiv, sigma_eiv)) * I + 2 * np.outer(sigma_eiv, sigma_eiv) - 2 * sigma_ei0 * sigma_eiv_cross


def error_dynamic(x, x_d, control):
    # 误差动力学
    p = x[:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    p_d = x_d[:3]
    v_d = x_d[3:6]
    u_d = x_d[6:9]
    q_d = x_d[9:13]
    w_d = x_d[13:16]
    tau_d = x_d[16:19]

    u = control[:3]
    tau = control[3:]

    p_e = p - p_d
    v_e = v - v_d

    q_d_inv = quaternion_conjugate(q_d)
    q_e = quaternion_multiply(q_d_inv, q)
    C_qe = rotation_matrix_from_quaternion(q_e)
    w_e = w - np.dot(C_qe, w_d)
    x_e = np.concatenate((p_e, v_e, q_e, w_e))

    J = np.diag([0.4167, 0.4167, 0.4167])

    d_p_e = v_e
    d_v_e = 0.1 * u - u_d  # m=10kg

    w_e_e = np.array([0, w[0], w[1], w[2]])
    d_q_e = 0.5 * quaternion_multiply(q_e, w_e_e)

    w_x = skew_symmetric(w)
    w_e_x = skew_symmetric(w_e)
    d_w_e = np.linalg.inv(J) @ (-w_x @ J @ w + tau) - C_qe @ np.clip(tau_d, -0.5, 0.5) + w_e_x @ C_qe @ w_d

    d_x_e = np.concatenate((d_p_e, d_v_e, d_q_e, d_w_e))

    return x_e, d_x_e


def game_dynamic(x_e, x_l_e, x_r_e, d_x_e, d_x_l_e, d_x_r_e, a, a_l, a_r):
    # 计算博弈状态和状态导数 即协同一致性误差和其导数，ei，ei_dot
    # 删掉q的标量
    game_x = (np.delete(x_e, 6) - np.delete(x_l_e, 6)) + (np.delete(x_e, 6) - np.delete(x_r_e, 6)) + np.delete(x_e, 6)

    d_game_x = a * np.delete(d_x_e, 6) + a_l * np.delete(d_x_l_e, 6) + a_r * np.delete(d_x_r_e, 6)

    return game_x, d_game_x


class ADP():
    # 自适应动态规划
    def __init__(self, dt):
        # 初始化
        self.dt = dt
        self.nn_dim = 18
        self.weights = 500 * np.ones((self.nn_dim))
        self.state_dim = 12
        self.u_dim = 6
        self.Q = 100 * np.eye(self.state_dim)
        self.R = 0.1 * np.eye(self.u_dim)
        self.yeta = 0.01  # learning rate
        self.g = np.array(
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0], [0, 0.1, 0, 0, 0, 0],
             [0, 0, 0.1, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 2.3998, 0, 0],
             [0, 0, 0, 0, 2.3998, 0], [0, 0, 0, 0, 0, 2.3998]])

        self.nn_dim_h = 6
        self.weights_h = 500 * np.ones((self.nn_dim_h))

    def cost_function(self, x, u, u_l, u_r):
        # 代价函数
        r = x.T @ self.Q @ x + u @ self.R @ u + u_l @ self.R @ u_l + u_r @ self.R @ u_r

        return r

    # phi对向量x的梯度
    def dot_Phi(self, x):
        # 基函数导数
        jacobian = np.zeros((self.nn_dim, self.state_dim))

        # 对每个 phi 项逐项求偏导
        jacobian[0, 0] = x[0]  # ∂(x[0]^2)/∂x[0]
        jacobian[1, 1] = x[1]  # ∂(x[1]^2)/∂x[1]
        jacobian[2, 2] = x[2]  # ∂(x[2]^2)/∂x[2]
        jacobian[3, 3] = x[3]  # ∂(x[3]^2)/∂x[3]
        jacobian[4, 4] = x[4]  # ∂(x[4]^2)/∂x[4]
        jacobian[5, 5] = x[5]  # ∂(x[5]^2)/∂x[5]

        jacobian[6, 0] = x[3]  # ∂(x[3]*x[0])/∂x[0]
        jacobian[6, 3] = x[0]  # ∂(x[3]*x[0])/∂x[3]
        jacobian[7, 1] = x[4]  # ∂(x[4]*x[1])/∂x[1]
        jacobian[7, 4] = x[1]  # ∂(x[4]*x[1])/∂x[4]
        jacobian[8, 2] = x[5]  # ∂(x[5]*x[2])/∂x[2]
        jacobian[8, 5] = x[2]  # ∂(x[5]*x[2])/∂x[5]

        jacobian[9, 6] = x[6]  # ∂(x[6]^2)/∂x[6]
        jacobian[10, 7] = x[7]  # ∂(x[7]^2)/∂x[7]
        jacobian[11, 8] = x[8]  # ∂(x[8]^2)/∂x[8]
        jacobian[12, 9] = x[9]  # ∂(x[9]^2)/∂x[9]
        jacobian[13, 10] = x[10]  # ∂(x[10]^2)/∂x[10]
        jacobian[14, 11] = x[11]  # ∂(x[11]^2)/∂x[11]

        jacobian[15, 6] = x[9]  # ∂(x[9]*x[6])/∂x[6]
        jacobian[15, 9] = x[6]  # ∂(x[9]*x[6])/∂x[9]
        jacobian[16, 7] = x[10]  # ∂(x[10]*x[7])/∂x[7]
        jacobian[16, 10] = x[7]  # ∂(x[10]*x[7])/∂x[10]
        jacobian[17, 8] = x[11]  # ∂(x[11]*x[8])/∂x[8]
        jacobian[17, 11] = x[8]  # ∂(x[11]*x[8])/∂x[11]

        return jacobian

    def Phi(self, x):
        # 基函数，激活函数，前六项，耦合了卫星ij的ei状态变量二次型形式，后续三项，位置与速度交叉用于描述耦合关系；后九项为ei中姿态与角速度部分
        phi = np.array(
            [0.5 * x[0] ** 2, 0.5 * x[1] ** 2, 0.5 * x[2] ** 2, 0.5 * x[3] ** 2, 0.5 * x[4] ** 2, 0.5 * x[5] ** 2,
             x[3] * x[0], x[4] * x[1], x[5] * x[2],
             0.5 * x[6] ** 2, 0.5 * x[7] ** 2, 0.5 * x[8] ** 2, 0.5 * x[9] ** 2, 0.5 * x[10] ** 2, 0.5 * x[11] ** 2,
             x[9] * x[6], x[10] * x[7], x[11] * x[8]])

        return phi

    def update_weights(self, x, d_x, u, u_l, u_r):
        # 更新权重
        r = self.cost_function(x, u, u_l, u_r)
        d_phi = self.dot_Phi(x)
        e = r + self.weights.T @ d_phi @ d_x  # 拟合Hamilton函数误差，最小化这个误差 公式5-41？
        theta = d_phi @ d_x  # 公式5-43后半
        d_weights = - self.yeta * theta * e  # 梯度下降
        self.weights = self.weights + d_weights * self.dt
        phi = self.Phi(x)
        v = self.weights.T @ phi  # 值函数，拟合的这个

        return self.weights, v

    def NE(self, x):
        # Nash均衡策略
        d_phi = self.dot_Phi(x)
        u_star = - 0.5 * np.linalg.inv(self.R) @ self.g.T @ d_phi.T @ self.weights

        # 如果不限幅值会崩溃 这是个很严重的问题，可以加饱和函数
        u_star[:3] = np.clip(u_star[:3], -5, 5)
        u_star[3:] = np.clip(u_star[3:], -0.5, 0.5)

        return u_star

    def cost_function_h(self, h_e, u_1, u_2, u_3, u_4):
        # 代价函数
        r = h_e.T @ self.Q @ h_e + u_1 @ self.R @ u_1 + u_2 @ self.R @ u_2 + u_3 @ self.R @ u_3 + u_4 @ self.R @ u_4

        return r

    def Phi_h(self, h):
        phi = np.array([0.5 * h[0] ** 2, 0.5 * h[1] ** 2, 0.5 * h[2] ** 2, h[0] * h[1], h[1] * h[2], h[2] * h[3]])

        return phi

    def dot_Phi_h(self, h):
        d_phi = np.zeros((6, 3))

        d_phi[0, 0] = h[0]
        d_phi[1, 1] = h[1]
        d_phi[2, 2] = h[2]
        d_phi[3, 0] = h[1]
        d_phi[3, 1] = h[0]
        d_phi[4, 0] = h[2]
        d_phi[4, 2] = h[0]
        d_phi[5, 1] = h[2]
        d_phi[5, 2] = h[1]

        return d_phi

    def update_weights_h(self, h_e, d_h_e, u1, u2, u3, u4):
        # 更新权重
        r = self.cost_function_h(h_e, u1, u2, u3, u4)
        d_phi = self.dot_Phi_h(h_e)
        e = r + self.weights_h.T @ d_phi @ d_h_e

        theta = d_phi @ d_h_e  # 公式5-43后半
        d_weights = - self.yeta * theta * e  # 梯度下降

        self.weights = self.weights + d_weights * self.dt
        phi = self.Phi(h_e)
        v = self.weights.T @ phi  # 值函数，拟合的这个

        return self.weights_h, v

    def NE_h(self, h_e, g_1, g_2, g_3, g_4):
        # Nash均衡策略
        d_phi = self.dot_Phi_h(h_e)
        u_1 = - 0.5 * np.linalg.inv(self.R) @ g_1.T @ d_phi.T @ self.weights_h
        u_2 = - 0.5 * np.linalg.inv(self.R) @ g_2.T @ d_phi.T @ self.weights_h
        u_3 = - 0.5 * np.linalg.inv(self.R) @ g_3.T @ d_phi.T @ self.weights_h
        u_4 = - 0.5 * np.linalg.inv(self.R) @ g_4.T @ d_phi.T @ self.weights_h

        # 如果不限幅值会崩溃 这是个很严重的问题，可以加饱和函数
        u_1[:3] = np.clip(u_1[:3], -5, 5)
        u_2[:3] = np.clip(u_2[:3], -5, 5)
        u_3[:3] = np.clip(u_3[:3], -5, 5)
        u_4[:3] = np.clip(u_4[:3], -5, 5)
        # u_1[3:] = np.clip(u_star[3:], -0.5, 0.5)

        return u_1, u_2, u_3, u_4