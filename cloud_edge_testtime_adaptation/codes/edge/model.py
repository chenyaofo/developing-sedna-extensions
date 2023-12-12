import logging

import os
import torch
import torchvision.models as models

from typing import Dict, List

from sedna.common.config import Context

from utils import download_file_to_temp

LOG = logging.getLogger(__name__)

os.environ['BACKEND_TYPE'] = 'TORCH'


def mobilenet_v2():
    return models.mobilenet_v2()


class Estimator:
    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        self.model = None
        self.infer_device = Context.get_parameters('infer_device')

    def load(self, model_url=""):
        local_model_path = download_file_to_temp(model_url)
        LOG.info(
            f"Load model from local path ({local_model_path}) | remote path ({model_url})")
        self.model = mobilenet_v2()
        self.model.load_state_dict(torch.load(local_model_path))
        self.model.to(device=self.infer_device).eval()

    def update_params(self, params_dict: Dict[str, List[float]]):
        param_device = next(self.model.parameters()).device
        param_dtype = next(self.model.parameters()).dtype

        bn_params_dict = {name: torch.tensor(params, device=param_device, dtype=param_dtype)
                          for name, params in params_dict.items()}
        self.model.load_state_dict(bn_params_dict, strict=False)

    def predict(self, data, **kwargs):
        # for the edge, just vanilla inference is required
        inputs = torch.tensor(data).to(device=self.infer_device)
        with torch.no_grad():
            outputs = self.model(inputs)
            return outputs.tolist()
