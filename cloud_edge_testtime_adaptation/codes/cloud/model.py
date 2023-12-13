import logging

import os
import torch
import torch.nn as nn
import torchvision.models as models

from typing import Dict, List

from sedna.common.config import Context

from utils import download_file_to_temp

from eta_core import *

LOG = logging.getLogger(__name__)

os.environ['BACKEND_TYPE'] = 'TORCH'


def mobilenet_v2_tta():
    return ETAModel(
        model=models.mobilenet_v2(),
        ent_high_margin=Context.get_parameters("ent_threshold"),
        tta_lr=Context.get_parameters("tta_lr")
    )


class ETAModel:
    def __init__(
        self,
        model: nn.Module,
        ent_high_margin: float,
        tta_lr: float,
        is_tbr: bool = True
    ) -> None:

        self.IS_FILTERED = True

        self.ent_high_margin = ent_high_margin
        self.model = model
        self.tta_lr = tta_lr

        assert isinstance(is_tbr, bool)
        self.is_tbr = is_tbr

        self.model.train()

        prepare_model_for_tent_tta(self.model, is_tbr=self.is_tbr)

        if self.is_tbr:
            replace_bn_forward_with(
                model=self.model,
                fn=tbr_bn_forward_impl
            )

        self.tta_optimizer = optim.SGD(
            params=[p for name, p in self.model.named_parameters() if p.requires_grad],
            lr=tta_lr,
            momentum=0.9
        )

        self.loss_fn = functools.partial(entropy, reduction="mean")

    def __call__(self, filtered_samples: torch.Tensor):
        set_bn_training_mode(self.model, mode=True)
        filtered_outputs = self.model(filtered_samples)
        ent: torch.Tensor = entropy(filtered_outputs, reduction="none")
        if self.ent_high_margin is not None:
            coeff = 1 / (torch.exp(ent.clone().detach() - self.ent_high_margin))
        else:
            coeff = 1
        loss = ent.mul(coeff).mean()

        self.tta_optimizer.zero_grad()
        loss.backward()
        self.tta_optimizer.step()
    
    def load_state_dict(self, *args, **kargs):
        return self.model.load_state_dict(*args, **kargs)

    def to(self, *args, **kargs):
        return self.model.to(*args, **kargs)

    def eval(self, *args, **kargs):
        return self.model.eval(*args, **kargs)

class Estimator:
    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        self.tta_model = None
        self.infer_device = Context.get_parameters('infer_device')

    def load(self, model_url=""):
        local_model_path = download_file_to_temp(model_url)
        LOG.info(
            f"Load model from local path ({local_model_path}) | remote path ({model_url})")
        self.tta_model = mobilenet_v2_tta()
        self.tta_model.load_state_dict(torch.load(local_model_path))
        self.tta_model.to(device=self.infer_device)
        self.tta_model.eval()

    def update_params(self, params_dict: Dict[str, List[float]]):
        param_device = next(self.tta_model.parameters()).device
        param_dtype = next(self.tta_model.parameters()).dtype

        bn_params_dict = {name: torch.tensor(params, device=param_device, dtype=param_dtype)
                          for name, params in params_dict.items()}
        self.tta_model.load_state_dict(bn_params_dict, strict=False)

    def predict(self, data, **kwargs):
        # for the edge, just vanilla inference is required
        inputs = torch.tensor(data).to(device=self.infer_device)

        outputs = self.tta_model(inputs)
        return outputs.tolist()
