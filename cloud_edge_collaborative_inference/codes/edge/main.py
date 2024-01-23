import time
import copy
import logging

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

from sedna.common.config import Context
from sedna.common.file_ops import FileOps
from sedna.core.joint_inference import JointInference

from model import Estimator
from custom import SplitInference
from utils import list_images_in_s3_path, download_file_to_temp

LOG = logging.getLogger(__name__)


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
    inference_instance = SplitInference(estimator=Estimator)
    edge_cloud_times = []
    cloud_times = []
    edge_times = []
    upload_times = []
    cloud_down_times = []
    right_num = 0
    debug = 1
    flag = 0
    for s3_image_path in list_images_in_s3_path(root):
        image_path = download_file_to_temp(s3_image_path)
        label = image_path.split('/')[-1].split('_')[0]
        img_rgb = preprocess(image_path)
        time_start = time.time() 
        _, cloud_result,cloud_use_time, edge_use_time,upload_time,cloud_down_time = (
            inference_instance.inference(img_rgb)
        )
        res = np.argmax(np.array(cloud_result[0]))
        time_end= time.time() 
        edge_cloud_time = time_end - time_start
        edge_cloud_times.append(edge_cloud_time)
        cloud_times.append(cloud_use_time)
        edge_times.append(edge_use_time)
        upload_times.append(upload_time)
        cloud_down_times.append(cloud_down_time)
        if(res==int(label)):
            right_num+=1
        if(debug):
            if(flag==200):
                break
            flag+=1
    print(f'''云边协同平均推理时间：{np.array(edge_cloud_times).mean()},
            云端平均推理时间：{np.array(cloud_times).mean()},
            边端平均推理时间：{np.array(edge_times).mean()},
            平均上传时间：{np.array(upload_times).mean()},
            平均下发时间：{np.array(cloud_down_times).mean()},
            准确率：{right_num/5000}''')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Entering sleep mode to allow log review...")
        time.sleep(3600)