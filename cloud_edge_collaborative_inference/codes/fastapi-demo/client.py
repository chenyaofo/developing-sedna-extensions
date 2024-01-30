# 客户端代码 (client.py)

import requests
import numpy as np
import base64
import time
import sys
import pickle
from pympler import asizeof
import sys


# file = {
#         "file": open("./file.txt", "rb")
#         }
# print(asizeof.asizeof(file))
# print(sys.getsizeof(file))

np_array = np.random.rand(1024,56,56).astype(np.float32)
# np.save('tmp.npy',np_array)
# print(asizeof.asizeof(np_array))
# print(sys.getsizeof(np_array))

encoded_data = np_array.tolist()

import json
serialized_data = json.dumps(encoded_data)
size_in_bytes = len(serialized_data.encode('utf-8'))
# print('size_in_bytes',size_in_bytes)

# with open("encoded_data.pkl", "wb") as file:
#     pickle.dump(encoded_data, file)
# print(asizeof.asizeof(encoded_data))
# print(sys.getsizeof(encoded_data))
time_start = time.time() 
# response = requests.post("http://127.0.0.1:8000/process-vector/", json={"data": encoded_data})
response = requests.post("http://127.0.0.1:8000/process-vector/", json={"data": encoded_data})
time_stop = time.time() 
if response.headers.get('Content-Type') == 'application/json':
    # 如果响应是JSON格式，使用.json()解析并打印
    print("Response JSON:")
    print(response.json())
else:
    # 否则，直接打印文本内容
    print("Response Text:")
    print(response.text)
print(time_stop-time_start)
