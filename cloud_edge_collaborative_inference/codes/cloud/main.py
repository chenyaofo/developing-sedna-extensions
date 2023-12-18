import time
from sedna.core.joint_inference import BigModelService

from patch import sedna_patch
from model import Estimator


def run():
    sedna_patch()
    inference_instance = BigModelService(estimator=Estimator)
    inference_instance.start()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Entering sleep mode to allow log review...")
        time.sleep(3600)