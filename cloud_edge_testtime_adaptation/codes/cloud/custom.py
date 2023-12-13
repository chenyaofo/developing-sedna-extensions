import torch.nn as nn
from sedna.core.joint_inference import BigModelService

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.responses import JSONResponse

from sedna.service.server.inference import InferenceItem, ServePredictResult, InferenceServer

from typing import List, Dict, Optional

TTA_BUFFER_LIMIT = 32


def bn_state_dicts(model):
    raw_model = model.estimator.tta_model.model
    full_state_dict = model.state_dict()

    # 初始化一个空的字典来存储BN层的state_dict
    bn_state_dict = {}

    # 遍历state_dict中的所有键值对
    for key, value in full_state_dict.items():
        # 检查键名是否包含“bn”（这假设BN层的名称中包含“bn”）
        if 'bn' in key:
            # 将BN层的参数添加到字典中
            bn_state_dict[key] = value.tolist()
    
    return bn_state_dict
    



class TTAUpdatedParamsResult(BaseModel):  # pylint: disable=too-few-public-methods
    params: Dict[str, List]


class TTAInferenceServer(InferenceServer):  # pylint: disable=too-many-arguments
    def start(self):
        self.buffer = []
        return self.run(self.app)

    def predict(self, data: InferenceItem):
        self.buffer.append(data)
        if len(self.buffer) < TTA_BUFFER_LIMIT:
            # in this case, we do not perform TTA since the buffer is not full
            # thus the params are not updated
            return TTAUpdatedParamsResult(params=None)
        else:
            self.model.inference(self.buffer, post_process=data.callback)
            self.buffer.clear()
            return TTAUpdatedParamsResult(params=bn_state_dicts(self.model))


class TTABigModelService(BigModelService):
    def start(self):
        if self.config.model_url.startswith("s3:"):
            self.estimator.load(self.config.model_url)
        else:
            raise FileExistsError(f"{self.config.model_url} is wrong")
        app_server = TTAInferenceServer(model=self, servername=self.job_name,
                                        host=self.local_ip, http_port=self.port)
        app_server.start()