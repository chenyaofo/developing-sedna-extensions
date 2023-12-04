import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2Shallow(nn.Module):
    def __init__(self, original_model, split_layer):
        super(MobileNetV2Shallow, self).__init__()
        self.features = nn.Sequential(*list(original_model.features[:split_layer]))

    def forward(self, x):
        x = self.features(x)
        return x

class MobileNetV2Deep(nn.Module):
    def __init__(self, original_model, split_layer):
        super(MobileNetV2Deep, self).__init__()
        self.features = nn.Sequential(*list(original_model.features[split_layer:]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_mobile_net_v2_parts(original_model, split_layer):
    """
    Splits a MobileNetV2 model into shallow and deep parts.

    Parameters:
    original_model (torch.nn.Module): The original MobileNetV2 model.
    split_layer (int): The layer index at which to split the model.

    Returns:
    MobileNetV2Shallow, MobileNetV2Deep: The shallow and deep parts of the model.
    """
    shallow_part = MobileNetV2Shallow(original_model, split_layer)
    deep_part = MobileNetV2Deep(original_model, split_layer)
    return shallow_part, deep_part

# Load the pretrained MobileNetV2 model
model = models.mobilenet_v2(pretrained=True)

model.eval()

with torch.no_grad():
    shallow_model, deep_model = create_mobile_net_v2_parts(model, 14)

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)

    # Convert to TorchScript
    shallow_script = torch.jit.trace(shallow_model, dummy_input)
    intermediate_features = shallow_model(dummy_input)
    deep_script = torch.jit.trace(deep_model, intermediate_features)

    # Save the models
    shallow_script.save("mobilenet_v2_shallow.pts")
    deep_script.save("mobilenet_v2_deep.pts")
