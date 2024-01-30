import socket
import numpy as np
import json
import time
# 创建 socket 对象
socket_client = socket.socket()
socket_client.connect(("localhost", 8888))

while True:
    # 用户输入第一维度大小
    first_dim = int(input("Enter the size of the first dimension of the array: "))

    # 定义数组维度并生成数组
    dimensions = (first_dim, 56, 56)
    np_array = np.random.rand(*dimensions).astype(np.float32)

    # 发送数组维度信息
    dim_info = json.dumps(dimensions)
    socket_client.send(dim_info.encode())

    # 等待确认
    ack = socket_client.recv(1024)
    if ack.decode() != "ACK":
        print("No ACK received")
        break

    # 发送数组数据
    time_start = time.time()
    binary_data = np_array.tobytes()
    socket_client.sendall(binary_data)

    # 接收响应
    data = socket_client.recv(1024).decode("UTF-8")
    print(f"服务器回复的消息为：{data}")
    time_end = time.time()-time_start
    print('所用时间为：',time_end)
# 关闭连接
socket_client.close()
