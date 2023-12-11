import torch
import torch.nn as nn
import torchvision.models as models


model = models.mobilenet_v2(pretrained=True)

model.eval()

state_dicts = model.state_dict()

torch.save(state_dicts, "mobilenet_v2.pt")
