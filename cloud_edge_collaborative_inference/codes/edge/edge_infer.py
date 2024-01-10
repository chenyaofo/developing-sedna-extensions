import torch
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import logging
from sedna.common.config import Context
from utils import list_images_in_s3_path, download_file_to_temp
import time 
import torchvision.models as models

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def preprocess(item):
    f = transforms.Compose([
        transforms.Lambda(lambda image_path: pil_loader(image_path)),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda tensor: tensor.unsqueeze(0))
    ])
    return f(item)
def main():
    logging.basicConfig(level=logging.INFO)
    root = Context.get_parameters("dataset")
    # root = 's3://cloud-edge/imagenet1000'
    model = models.mobilenet_v2(pretrained=True).to(device='cuda')
    model.eval()
    times = []
    right_num = 0
    for s3_image_path in list_images_in_s3_path(root):
        image_path = download_file_to_temp(s3_image_path)
        print('image_path',image_path)
        label = image_path.split('/')[-1].split('_')[0]
        img_rgb = preprocess(image_path)
        inputs = torch.tensor(img_rgb).to(device='cuda')
        time_start = time.time() 
        with torch.no_grad():
            edge_result = model(inputs)
            res = torch.argmax(edge_result)
        time_end = time.time() 
        time_sum = time_end - time_start
        times.append(time_sum)
        res = res.cpu().numpy()
        # print(res,label)
        if(res==int(label)):
            right_num+=1
        print(right_num)
    print(f'平均推理时间：{np.array(times).mean()},准确率：{right_num/5000}')

if __name__=='__main__':
    main()