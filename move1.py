# encoding:utf-8
import socket
# IP+PORT
HOST = "192.168.1.5"  # UR5的ip地址
PORT = 30003            # UR5的端口号30003（125HZ）

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.connect((HOST, PORT))
x, y, z = 252.3, 456.6, -232.1,
rx, ry, rz = 2.163, -3.051, -1.349,
# movel直线移动；p[x, y, z, rx, ry, rz] 与示教器相同的6D姿态
cmd_data = b"movel(p[252.3, 456.6, -232.1, 2.163, -3.051, -1.349,], a=0.1, v=0.1)\n"
server.send(cmd_data)

server.close()
