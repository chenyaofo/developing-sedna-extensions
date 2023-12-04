from sedna.core.joint_inference import BigModelService

from patch import sedna_patch
from model import Estimator


def run():
    sedna_patch()
    inference_instance = BigModelService(estimator=Estimator)
    inference_instance.start()


if __name__ == "__main__":
    run()
