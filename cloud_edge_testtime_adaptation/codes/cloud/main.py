import time
from custom import TTABigModelService
from model import Estimator


def run():
    inference_instance = TTABigModelService(estimator=Estimator)
    inference_instance.start()


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"An exception occurred: {e}")
        print("Entering sleep mode to allow log review...")
        time.sleep(3600)
