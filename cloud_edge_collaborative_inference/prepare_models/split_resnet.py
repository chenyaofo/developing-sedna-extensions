# import torch
# import torchvision.models as models

# # 加载预训练的ResNet152模型
# model = models.resnet152(pretrained=True)

# # 打印模型结构以确定切分点
# print(model)

# # 假设我们要在layer3和layer4之间切分模型
# # 创建包含从开始到layer3的模型
# model_part1 = torch.nn.Sequential(*(list(model.children())[:8]))

# # 创建包含从layer4到模型末尾的模型
# model_part2 = torch.nn.Sequential(*(list(model.children())[8:]))

# # 保存两个模型
# torch.save(model_part1, 'resnet152_part1.pth')
# torch.save(model_part2, 'resnet152_part2.pth')



import torch
import torch.nn as nn
import torchvision.models as models

class ResNet152Shallow(nn.Module):
    def __init__(self, original_model, split_layer):
        super(ResNet152Shallow, self).__init__()
        self.features = nn.Sequential(*(list(original_model.children())[:split_layer]))

    def forward(self, x):
        x = self.features(x)
        return x

class ResNet152Deep(nn.Module):
    def __init__(self, original_model, split_layer):
        super(ResNet152Deep, self).__init__()
        self.features = nn.Sequential(*(list(original_model.children())[split_layer:-1]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = list(original_model.children())[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_resnet_152_parts(original_model, split_layer):
    shallow_part = ResNet152Shallow(original_model, split_layer)
    deep_part = ResNet152Deep(original_model, split_layer)
    return shallow_part, deep_part

# Load the pretrained MobileNetV2 model
model = models.resnet152(pretrained=True)
model.eval()

with torch.no_grad():
    shallow_model, deep_model = create_resnet_152_parts(model, 6)

    # Dummy input for tracing
    dummy_input = torch.randn(1, 3, 224, 224)

    # Convert to TorchScript
    shallow_script = torch.jit.trace(shallow_model, dummy_input)
    intermediate_features = shallow_model(dummy_input)
    deep_script = torch.jit.trace(deep_model, intermediate_features)

    # Save the models
    shallow_script.save("resnet152_shallow.pts")
    deep_script.save("resnet152_deep.pts")
    print(model(dummy_input)[0][0])
    print(deep_model(intermediate_features)[0][0])