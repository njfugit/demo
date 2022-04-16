import cv2
from numpy import *
import demo.coordinate
import demo.photo
import socket
import torch
from demo.getBoxesAndDepth import return_value

obj_num = 100
depth_img, rgb_img = demo.photo.get_image(obj_num)
depth = cv2.imread('100depth.png', -1)

pix_x, pix_y = return_value()
point3d = demo.coordinate.Pixels2Cam1(pix_x, pix_y, depth[pix_y, pix_x])
photo_robot_tran = [559.88, -395.7, 45.33]
photo_robot_rotate = [0.165, -4.644, 0.18]
point2base = demo.coordinate.CamObject2Base(point3d, photo_robot_tran, photo_robot_rotate)


# 固定的方向——R
rota_rx, rota_ry, rota_rz = 0.067, 1.874, -0.003
move_r = torch.tensor([rota_rx, rota_ry, rota_rz])

# 旋转向量转旋转矩阵
R = demo.coordinate.Vec2Matrix(move_r)
move_R = torch.tensor(
        [[R[0, 0], R[0, 1], R[0, 2]], [R[1, 0], R[1, 1], R[1, 2]], [R[2, 0], R[2, 1], R[2, 2]], [0, 0, 0]])
swab2tcp = torch.tensor([-1.7635, -5.8617, 244.6087])
move_offset = torch.matmul(move_R, swab2tcp)


move_target = torch.tensor([point2base[0], point2base[1], point2base[2]])

move_actual_x = (move_target[0] - move_offset[0]).numpy()
move_actual_y = (move_target[1] - move_offset[1]).numpy()
move_actual_z = (move_target[2] - move_offset[2]).numpy()
print(move_actual_x)
print(move_actual_y)
print(move_actual_z)
#
# # 1.拍照位置（固定）
# # 2.进入口腔预备姿势（y、z值取目标位置的y、z值，固定旋转向量）
# # 3.进入口腔（旋转向量同3）
# # 4.旋转（正）
# # 5.旋转（负）
# # 6.退回预备姿势
# # 7.退回拍照位固置（定）
# photo_x, photo_y, photo_z, photo_rx, photo_ry, photo_rz = 0.6028, -0.25920, 0.02285, 0.172, 1.896, -0.063
#
#
# x, y, z = move_actual_x/1000, move_actual_y/1000, move_actual_z/1000
#
# send_data = '''def functionName():\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
# movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n''' % (photo_x, photo_y, photo_z, photo_rx, photo_ry, photo_rz,
#                                                       x - 0.15, y, z, rota_rx, rota_ry, rota_rz,
#                                                       x, y, z, rota_rx, rota_ry, rota_rz,
#                                                       x, y, z, rota_rx, rota_ry, rota_rz,
#                                                       x, y, z, rota_rx, rota_ry, rota_rz,
#                                                       x - 0.2, y, z, rota_rx, rota_ry, rota_rz,
#                                                       photo_x, photo_y, photo_z, photo_rx, photo_ry, photo_rz)
# send = send_data + '''end\n'''
# print(send)
#
# target_ip = ("1192.168.174.128", 30003)
# # 建立一个socket对象
# sk = socket.socket()
# sk.connect(target_ip)
#
# # 发送指令，并将字符串转变格式
# sk.send(send.encode('utf8'))
# sk.close()
#
#
#
#
