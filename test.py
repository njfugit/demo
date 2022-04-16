# import cv2
# from numpy import *
# import coordinate
# import photo
# import socket
# import torch
#
#
# obj_num = 100
# depth_img, rgb_img = photo.get_image(obj_num)
# depth = cv2.imread('biaoding/74depth.png', -1)
#
#
# # def depth2gray(depth):
# #     img_copy = depth
# #     max_v = img_copy.max()
# #     min_v = img_copy.min()
# #     img = (img_copy - min_v) / (max_v - min_v)
# #     img = img * 255
# #     return img
# #
# # depth_img = depth2gray(depth)
# # cv2.imshow('ddd', depth_img)
# # cv2.waitKey(10000)
# pix_x, pix_y = 267, 219
# point3d = coordinate.Pixels2Cam1(pix_x, pix_y, depth[pix_y, pix_x])
# print(point3d)
# photo_robot_tran = [458.39, -174.02, 44.53]
# photo_robot_rotate = [0.418, 1.808, -0.187]
# point2base = coordinate.CamObject2Base(point3d, photo_robot_tran, photo_robot_rotate)
# print(point2base)
#
#
#
#
#
#


from demo.getBoxesAndDepth import return_value

x, y = return_value()
print(x, y)