# 服务端代码 (server.py)

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import base64

class Vector(BaseModel):
    data: List  # Base64 编码的字符串

app = FastAPI()

@app.post("/process-vector/")
def process_vector(vector: Vector):
    print(np.array(vector.data).shape)
    return {"result": 1}
