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

from utils import list_images_in_s3_path, download_file_to_temp

LOG = logging.getLogger(__name__)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def preprocess(item):
    # note that these transforms are for ImageNet-C, which is not the same as ImageNet
    f = transforms.Compose([
        transforms.Lambda(lambda image_path: pil_loader(image_path)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return f(item)


def main():
    root = Context.get_parameters("dataset")
    inference_instance = JointInference(estimator=Estimator)

    for s3_image_path in list_images_in_s3_path(root):
        image_path = download_file_to_temp(s3_image_path)
        img_rgb = preprocess(image_path)
        is_hard_example, edge_result = (
            inference_instance.inference(img_rgb)
        )

        LOG.info(f"For image {s3_image_path} (is_hard_sample={is_hard_example}), inferred result is {edge_result}")


if __name__ == '__main__':
    main()
