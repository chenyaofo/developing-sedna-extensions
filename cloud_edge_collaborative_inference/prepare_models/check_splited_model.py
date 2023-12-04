import torch
import torch.nn as nn
import torchvision.models as models

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)
model.eval()
# print(model)

# Function to load a TorchScript model
def load_torchscript_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()  # Set the model to evaluation mode
    return model

shallow_model = load_torchscript_model("mobilenet_v2_shallow.pts")
deep_model = load_torchscript_model("mobilenet_v2_deep.pts")


with torch.no_grad():
    x = torch.rand(1,3,224,224)

    y1 = model(x)
    y2 = deep_model(shallow_model(x))

    print(f"The differences between torch scripts and the original model:",torch.max(torch.abs(y1-y2)))