# 服务端代码 (server.py)

from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import base64
from fastapi import FastAPI, Request

class Vector(BaseModel):
    data: List

app = FastAPI()

@app.post("/process-vector/")
def process_vector(request: Request,vector: Vector):
    # print(np.array(vector.data).shape)
    content_length = request.headers.get('content-length')
    if content_length:
        print(f"Content-Length: {content_length}")
    else:
        print("Content-Length header is missing")
    return {"传输数据大小为": float(content_length)/1024/1024}


@app.post("/get-vector/")
def process_vector():
    return {"result": 1}

