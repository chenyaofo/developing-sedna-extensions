# 客户端代码 (client.py)

import requests
import numpy as np
import base64
import time

np_array = np.random.rand(512, 28,28).astype(np.float32)

encoded_data = np_array.tolist()
time_start = time.time() 
response = requests.post("http://127.0.0.1:8000/process-vector/", json={"data": encoded_data})
time_stop = time.time() 
print(time_stop-time_start)
