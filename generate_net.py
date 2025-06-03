# 11*11 折叠 
# 卫星只有位置移动的三个自由度
import xml.etree.ElementTree as ET
import os
import numpy as np


def findpos(i, j, rows, cols, length):
    x = -10
    y = 0
    z = 0
    if j <= (cols + 1) / 2:
        y = length * ((cols + 1) / 2 - j)
    elif ((cols + 1) / 2 + 1) <= j <= cols:
        y = -length * (j - (cols + 1) / 2)

    if i <= (rows + 1) / 2:
        z = length * ((rows + 1) / 2 - i)
    elif ((rows + 1) / 2 + 1) <= i <= rows:
        z = -length * (i - (rows + 1) / 2)
    return x, y, z


def create_grid_model(rows, cols,file_name):
    
    # 网格长度
    length_mesh = 0.5
    delta = 0.01


    # 创建根元素
    root = ET.Element("mujoco")

    # 求解器
    ET.SubElement(root,"option",integrator="RK4", timestep="0.001", solver="PGS")

    # 导入地球背景
    ET.SubElement(root,"include",file="scene.xml")
    # 重力
    option =  ET.SubElement(root,"option",gravity="0 0 0")

    # 关闭所有碰撞检测
    ET.SubElement(option,"flag",contact="disable")
    
    # 分配内存
    #ET.SubElement(root,"size",memory="10G")
    
    # 绳子设置
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "tendon", limited="true", width="0.005", rgba="1 1 0 1",damping="0.07",stiffness="40")
    
    # 光线设置
    asset = ET.SubElement(root, "visual")
    ET.SubElement(asset, "global", azimuth="-90", elevation="-40")
    ET.SubElement(asset, "headlight", diffuse="1 1 1")

    # 创建worldbody元素
    worldbody = ET.SubElement(root, "worldbody")

    # 相机
    ET.SubElement(worldbody,"camera",name="closeup", pos="10 10 10", xyaxes="-0.4 0.9 0.000 -0.2 -0.1 0.9")
    
    # 定义四个MU
    num_MU = 4
    MU_pos = [f"-10 {length_mesh*5+0.25} {length_mesh*5+0.25}",f"-10 {-length_mesh*5-0.25} {length_mesh*5+0.25}",f"-10 {length_mesh*5+0.25} {-length_mesh*5-0.25}",f"-10 {-length_mesh*5-0.25} {-length_mesh*5-0.25}"]
    MU_site_pos = ["0 -0.25 -0.25","0 0.25 -0.25","0 -0.25 0.25","0 0.25 0.25"]

    for i in range(num_MU):
        node_name = f"MU_{i}"
        pos = MU_pos[i]
        site_pos = MU_site_pos[i]

        MU = ET.SubElement(worldbody, "body", name=node_name, pos=pos)
        ET.SubElement(MU, "joint", name=f"sat_{i}", type="free")
        # ET.SubElement(MU, "joint",name=f"x_{i}", type="slide",axis="1 0 0")
        # ET.SubElement(MU, "joint",name=f"y_{i}", type="slide",axis="0 1 0")
        # ET.SubElement(MU, "joint",name=f"z_{i}", type="slide",axis="0 0 1")
        ET.SubElement(MU, "inertial", pos="-10 0 0", diaginertia="0.4167 0.4167 0.4167", mass="10")
        ET.SubElement(MU, "geom", type="box", size=".25 .25 .25", material="satellite",mass="10")
        ET.SubElement(MU, "site",pos=site_pos,name=f"MU_{i}", type="sphere", size="0.0005")
    
    # 定义网子的节点
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                continue
            elif i == 0 and j == cols-1:
                continue
            elif i == rows-1 and j == 0:                   
                continue
            elif i == rows-1 and j == cols-1:
                break
            else:
                node_name = f"node_{i}_{j}"
                pos_x,pos_y,pos_z = findpos(i+1,j+1,rows,cols,length_mesh)
                pos = f"{pos_x} {pos_y} {pos_z}"

                node = ET.SubElement(worldbody, "body", name=node_name, pos=pos)
                ET.SubElement(node, "joint", type="free")
                ET.SubElement(node, "geom", type="sphere", size="0.0005", rgba="1 0 0 1",mass="0.0011")
                ET.SubElement(node, "site",name=f"node_{i}_{j}", type="sphere", size="0.0005")
    
    # 连接水平绳子
    tendon = ET.SubElement(root, "tendon")
    for i in range(rows):
        for j in range(cols-1):
            if i == 0 and j == 0:
                node1_name = f"MU_{0}"
                node2_name = f"node_{i}_{j+1}"
            elif i == 0 and j == cols-2:
                node1_name = f"node_{i}_{j}"
                node2_name = f"MU_{1}"
            elif i == rows-1 and j == 0:                   
                node1_name = f"MU_{2}"
                node2_name = f"node_{i}_{j+1}"
            elif i == rows-1 and j == cols-2:
                node1_name = f"node_{i}_{j}"
                node2_name = f"MU_{3}"
            else:
                node1_name = f"node_{i}_{j}"
                node2_name = f"node_{i}_{j+1}"
            spatial = ET.SubElement(tendon, "spatial", range=f"0 {length_mesh+delta}",springlength=f"0 {length_mesh}")
            ET.SubElement(spatial, "site", site=node1_name)
            ET.SubElement(spatial, "site", site=node2_name)

    # 连接垂直绳子
    for i in range(rows-1):
        for j in range(cols):
            if i == 0 and j == 0:
                node1_name = f"MU_{0}"
                node2_name = f"node_{i+1}_{j}"
            elif i == 0 and j == cols-1:
                node1_name = f"MU_{1}"
                node2_name = f"node_{i+1}_{j}"
            elif i == rows-2 and j == 0: 
                node1_name = f"node_{i}_{j}"                  
                node2_name = f"MU_{2}"
            elif i == rows-2 and j == cols-1:
                node1_name = f"node_{i}_{j}"
                node2_name = f"MU_{3}"
            else:
                node1_name = f"node_{i}_{j}"
                node2_name = f"node_{i+1}_{j}"
            spatial = ET.SubElement(tendon, "spatial", range=f"0 {length_mesh+delta}",springlength=f"0 {length_mesh}")
            ET.SubElement(spatial, "site", site=node1_name)
            ET.SubElement(spatial, "site", site=node2_name)
    

    # 加控制器
    u_max = 5
    t_max = 0.5
    actuator = ET.SubElement(root, "actuator")
    for i in range(num_MU):
        # for j in "xyz":
        #     ET.SubElement(actuator, "motor", name=f"move_{j}_{i}", joint=f"{j}_{i}", ctrlrange=f"-{u_max} {u_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"force_{i}_x", joint=f"sat_{i}", gear="1 0 0 0 0 0",
                      ctrlrange=f"-{u_max} {u_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"force_{i}_y", joint=f"sat_{i}", gear="0 1 0 0 0 0",
                      ctrlrange=f"-{u_max} {u_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"force_{i}_z", joint=f"sat_{i}", gear="0 0 1 0 0 0",
                      ctrlrange=f"-{u_max} {u_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"torque_{i}_x", joint=f"sat_{i}", gear="0 0 0 1 0 0",
                     ctrlrange=f"-{t_max} {t_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"torque_{i}_y", joint=f"sat_{i}", gear="0 0 0 0 1 0",
                     ctrlrange=f"-{t_max} {t_max}", ctrllimited="true")
        ET.SubElement(actuator, "motor", name=f"torque_{i}_z", joint=f"sat_{i}", gear="0 0 0 0 0 1",
                     ctrlrange=f"-{t_max} {t_max}", ctrllimited="true")



    # 传感器
    sensor = ET.SubElement(root,"sensor")
    for i in range(num_MU):
            ET.SubElement(sensor, "framelinvel", name=f"vel_{i}", objtype="body", objname=f"MU_{i}")  # 速度传感器
            ET.SubElement(sensor, "force", name=f"force_{i}", site=f"MU_{i}")  # 力传感器
            ET.SubElement(sensor, "frameangvel", name=f"angvel_{i}", objtype="body", objname=f"MU_{i}")  # 角速度传感器




    # 创建MJCF模型
    mjcf_tree = ET.ElementTree(root)

    # 保存到文件
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    mjcf_tree.write(file_name, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    rows = 11
    cols = 11
    file_name = os.path.join('TSNR', f"net_{rows}_{cols}_h.mjcf")
    create_grid_model(rows, cols,file_name)
