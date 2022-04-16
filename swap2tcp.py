# import cv2
# from numpy import *
# import coordinate
# import photo
# import socket
# import torch
#
#
# # 1.拍照位置（固定）
# # 2.进入口腔（旋转向量同1）
# # 3.旋转
# # 4.退回拍照位置（固定）
#
# send_data = '''def functionName():\n
# movej(p[0.5194, -0.280, 0.0585, 0.057, 1.962, -0.094],a=0.1, v=0.3, r=0)\n
# movej(p[0.711, -0.323, -0.022, 0.04, 1.692, -0.071],a=0.1, v=0.2, r=0)\n
# movej(p[0.737, -0.319, -0.029, 0.032, 1.716, -0.053],a=0.1, v=0.2, r=0)\n
# movej(p[0.739, -0.319, -0.029, 0.563, 1.64, 0.409],a=0.1, v=0.2, r=0)\n
# movej(p[0.737, -0.319, -0.029, 0.032, 1.716, -0.053],a=0.1, v=0.2, r=0)\n
# movej(p[0.711, -0.323, -0.022, 0.04, 1.692, -0.071],a=0.1, v=0.2, r=0)\n
# movej(p[0.5194, -0.280, 0.0585, 0.057, 1.962, -0.094],a=0.2, v=0.2, r=0)\n'''
# send = send_data + '''end\n'''
# print(send)
#
# target_ip = ("192.168.1.5", 30003)
# # 建立一个socket对象
# sk = socket.socket()
# sk.connect(target_ip)
#
# # 发送指令，并将字符串转变格式
# sk.send(send.encode('utf8'))
# sk.close()
#

# 求swab2tcp
import cv2
from numpy import *
import coordinate
import photo
import socket
import torch


obj_num = 98
# depth_img, rgb_img = photo.get_image(obj_num)
depth = cv2.imread('biaoding/98depth.png', -1)


# def depth2gray(depth):
#     img_copy = depth
#     max_v = img_copy.max()
#     min_v = img_copy.min()
#     img = (img_copy - min_v) / (max_v - min_v)
#     img = img * 255
#     return img
#
# depth_img = depth2gray(depth)
# cv2.imshow('ddd', depth_img)
# cv2.waitKey(10000)

pix_x, pix_y = 435, 466
point3d = coordinate.Pixels2Cam1(pix_x, pix_y, depth[pix_y, pix_x])

photo_robot_tran = [559.88, -395.7, 45.33]
photo_robot_rotate = [0.165, -4.644, 0.18]

point2base = coordinate.swap2Tcp(pix_x, pix_y, depth)
print(point2base)