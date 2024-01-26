import abc

from sedna.common.class_factory import ClassFactory, ClassType
from sedna.algorithms.hard_example_mining.hard_example_mining import BaseFilter

@ClassFactory.register(ClassType.HEM, alias="AllUpload")
class AllUploadFilter(BaseFilter, abc.ABC):
    def __init__(self, **kwargs):
        pass

    def __call__(self, infer_result=None):
        return True