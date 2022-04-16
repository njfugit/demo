# encoding:utf-8

import cv2
from numpy import *
import demo.coordinate
import demo.photo
import socket
import torch
#from demo.getBoxesAndDepth import return_value



#pix_x, pix_y = return_value()
#point3d = demo.coordinate.Pixels2Cam1(pix_x, pix_y, depth[pix_y, pix_x])
#photo_robot_tran = [559.88, -395.7, 45.33]
#photo_robot_rotate = [0.165, -4.644, 0.18]
#point2base = demo.coordinate.CamObject2Base(point3d, photo_robot_tran, photo_robot_rotate)


# 固定的方向——R
#rota_rx, rota_ry, rota_rz = 0.067, 1.874, -0.003
#move_r = torch.tensor([rota_rx, rota_ry, rota_rz])

# 旋转向量转旋转矩阵
#R = demo.coordinate.Vec2Matrix(move_r)
#move_R = torch.tensor(
        #[[R[0, 0], R[0, 1], R[0, 2]], [R[1, 0], R[1, 1], R[1, 2]], [R[2, 0], R[2, 1], R[2, 2]], [0, 0, 0]])
#swab2tcp = torch.tensor([-1.7635, -5.8617, 244.6087])
#move_offset = torch.matmul(move_R, swab2tcp)
photo_x, photo_y, photo_z, photo_rx, photo_ry, photo_rz = 300.62, 476.25, -308.86, 2.625, -2.747, -1.476,
#
#
# x, y, z = move_actual_x/1000, move_actual_y/1000, move_actual_z/1000

send_data = '''def functionName():\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n
movej(p[%f,%f,%f,%f,%f,%f],a=0.1, v=0.1, r=0)\n''' % (300.62, 476.25, -308.86, 2.625, -2.747, -1.476,
                                                       311.28, 527.44, -356.72, 2.645, -2.718, -1.502,
                                                       324.58, 620.37, -400.48, 2.700, -2.622, -1.625,
                                                       324.58, 620.37, -400.48, 1.045, -2.934, -1.992,
                                                       324.58, 620.37, -400.48, 2.700, -2.622, -1.625,
                                                       311.28, 527.44, -356.72, 2.645, -2.718, -1.502,
                                                       300.62, 476.25, -308.86, 2.625, -2.747, -1.476,)
send = send_data + '''end\n'''
print(send)

target_ip = ("192.168.1.5", 30003)
# 建立一个socket对象
sk = socket.socket()
sk.connect(target_ip)

# 发送指令，并将字符串转变格式
sk.send(send.encode('utf8'))
sk.close()
