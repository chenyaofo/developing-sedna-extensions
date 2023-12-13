from custom import TTABigModelService
from model import Estimator


def run():
    inference_instance = TTABigModelService(estimator=Estimator)
    inference_instance.start()


if __name__ == "__main__":
    run()
