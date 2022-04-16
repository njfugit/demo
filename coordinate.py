import cv2
from numpy import *
import torch

# depth为深度值
def Pixels2Cam1(x, y, depth):
    camera_factor = 1000
    camera_cx = 321.084
    camera_cy = 238.077
    camera_fx = 609.441
    camera_fy = 607.96
    rz = depth / camera_factor
    rx = (x - camera_cx) * rz / camera_fx
    ry = (y - camera_cy) * rz / camera_fy
    point3d = (rx, ry, rz)
    return point3d


# 矩阵拼接
def R_T2HomogeneousMatrix(R, T):
    R1 = torch.tensor(
        [[R[0, 0], R[0, 1], R[0, 2]], [R[1, 0], R[1, 1], R[1, 2]], [R[2, 0], R[2, 1], R[2, 2]], [0, 0, 0]])
    T1 = torch.tensor([[T[0]], [T[1]], [T[2]], [1]])  # 4*1
    HomoMtr = torch.cat((R1, T1), 1)
    return HomoMtr


# 旋转向量转化成旋转矩阵的函数
def Vec2Matrix(n):
    a = n.numpy()
    R = cv2.Rodrigues(a)
    return R[0]


# 欧拉角转化成旋转矩阵的函数
def eular2Matrix2(n):
    # R = torch.zeros(3, 3, dtype=torch.long)

    theta_x = n[0] * (3.1415926 / 180)
    theta_y = n[1] * (3.1415926 / 180)
    theta_z = n[2] * (3.1415926 / 180)

    Rx = torch.tensor([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)], [0, sin(theta_x), cos(theta_x)]])
    Ry = torch.tensor([[cos(theta_y), 0, sin(theta_y)], [0, 1, 0], [-sin(theta_y), 0, cos(theta_y)]])
    Rz = torch.tensor([[cos(theta_z), -sin(theta_z), 0], [sin(theta_z), cos(theta_z), 0], [0, 0, 1]])
    # R = Rz * Ry * Rx
    R = torch.matmul(Rz, Ry)
    R = torch.matmul(R, Rx)

    return R


# cam_coord:Pixels2Cam返回的结果
# robot_trans:Robot的x,y,z
# robot_rotate:Robot的Rx,Ry,Rz
def CamObject2Base(cam_coord, robot_trans_vec, robot_rotate_vec):
    cam2flange = torch.tensor([[0.02295552830003134, 0.9995017242488096, -0.02166441654240945, -78.94841417841896],
                               [-0.9997364566173423, 0.02294478218985208, -0.0007445001415335786, 32.8049371645021],
                               [-0.0002470438563304863, 0.02167579742285891, 0.9997650217803262, 26.91415148250717],
                               [0, 0, 0, 1]])
    # 欧拉角转化成旋转矩阵的函数
    # a1 = float(robot_trans_vec[0])
    # a2 = float(robot_trans_vec[1])
    # a3 = float(robot_trans_vec[2])
    robot_trans = torch.tensor([float(robot_trans_vec[0]), float(robot_trans_vec[1]), float(robot_trans_vec[2])])  #
    robot_rotate = torch.tensor([float(robot_rotate_vec[0]), float(robot_rotate_vec[1]), float(robot_rotate_vec[2])])
    base_rotate = Vec2Matrix(robot_rotate)
    base_trans = robot_trans
    # 矩阵拼接
    gripper2base_matrix = R_T2HomogeneousMatrix(base_rotate, base_trans)
    # print(gripper2base_matrix)

    cam2cal_matrix = torch.tensor(
        [[float(cam_coord[0]) * 1000], [float(cam_coord[1]) * 1000], [float(cam_coord[2]) * 1000], [1]])

    # camera到基座标
    b = torch.matmul(gripper2base_matrix, cam2flange)
    # print(b)
    c = torch.matmul(b, cam2cal_matrix)
    c = c.numpy()

    return c[0, 0], c[1, 0], c[2, 0]


def swap2Tcp(pix_x, pix_y, depth):

    cam2flange = torch.tensor([[0.02295552830003134, 0.9995017242488096, -0.02166441654240945, -78.94841417841896],
                               [-0.9997364566173423, 0.02294478218985208, -0.0007445001415335786, 32.8049371645021],
                               [-0.0002470438563304863, 0.02167579742285891, 0.9997650217803262, 26.91415148250717],
                               [0, 0, 0, 1]])
    cam_coord = Pixels2Cam1(pix_x, pix_y, depth[pix_y, pix_x])
    cam2cal_matrix = torch.tensor(
        [[float(cam_coord[0]) * 1000], [float(cam_coord[1]) * 1000], [float(cam_coord[2]) * 1000], [1]])

    swab2tcp = torch.matmul(cam2flange, cam2cal_matrix)
    print('swab2tcp', swab2tcp)
    return swab2tcp[0, 0], swab2tcp[1, 0], swab2tcp[2, 0]