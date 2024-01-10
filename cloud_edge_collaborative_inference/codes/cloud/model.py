import logging
import os
import torch

from sedna.common.config import Context

from utils import download_file_to_temp

LOG = logging.getLogger(__name__)
os.environ['BACKEND_TYPE'] = 'TORCH'

class Estimator:
    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        self.model = None
        self.infer_device = Context.get_parameters('infer_device')

    def load(self, model_url=""):
        model_url = os.getenv("MODEL_URL")
        # local_model_path = download_file_to_temp(model_url)
        LOG.info(
            f"Load model from local path ({model_url}) | remote path ({model_url})")
        self.model = torch.jit.load(model_url).to(device=self.infer_device).eval()

    def predict(self, data, **kwargs):
        inputs = torch.tensor(data).to(device=self.infer_device)
        with torch.no_grad():
            outputs = self.model(inputs)
            return outputs.tolist()
