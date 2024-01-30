import socket
import numpy as np
import json
import numbers

def is_number(var):
    return isinstance(var, numbers.Number)
# 创建 socket 对象
socket_server = socket.socket()
socket_server.bind(("localhost", 8888))
socket_server.listen(1)

print("等待客户端连接...")
conn, address = socket_server.accept()
print(f"接收到的客户端连接信息为{address}")

while True:
    # 接收维度信息
    dim_info = conn.recv(1024).decode()
    if not dim_info:
        print("No dimension info received")
        continue
    # print(type(dim_info))
    dimensions = json.loads(dim_info)
    expected_size = np.prod(dimensions) * 4  # float32 占 4 字节

    # 发送确认信号
    conn.send(b"ACK")

    # 接收数据
    received_data = bytearray()
    while len(received_data) < expected_size:
        packet = conn.recv(expected_size - len(received_data))
        if not packet:
            break
        received_data.extend(packet)

    # 转换数据
    np_array = np.frombuffer(received_data, dtype=np.float32).reshape(*dimensions)
    print(f"接收到的 numpy 数组: {np_array.shape}")

    # 计算数据大小并回复
    data_size_mb = len(received_data) / (1024 * 1024)
    response = f"Received data size: {data_size_mb:.2f} MB"
    conn.send(response.encode())

# 关闭连接
conn.close()
socket_server.close()
