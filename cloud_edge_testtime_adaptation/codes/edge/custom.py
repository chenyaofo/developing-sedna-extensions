import os
from copy import deepcopy

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase
from sedna.core.joint_inference import JointInference

class TTAJointInference(JointInference):
    # here we modify the inference function
    def inference(self, data=None, post_process=None, **kwargs):
        callback_func = None
        if callable(post_process):
            callback_func = post_process
        elif post_process is not None:
            callback_func = ClassFactory.get_cls(
                ClassType.CALLBACK, post_process)

        res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        if callback_func:
            res = callback_func(res)

        self.lc_reporter.update_for_edge_inference()

        is_hard_example = False
        cloud_result = None

        if self.hard_example_mining_algorithm:
            is_hard_example = self.hard_example_mining_algorithm(res)
            if is_hard_example:
                try:
                    cloud_data = self.cloud.inference(
                        data.tolist(), post_process=post_process, **kwargs)
                    # all we modify is the following codes
                    # ------------------------------------------
                    updated_params = cloud_data["params"]
                    if updated_params:
                        self.estimator.update_params(updated_params)
                        self.log.info("Update params from the cloud")
                    # ------------------------------------------
                except Exception as err:
                    self.log.error(f"get cloud result error: {err}")
                else:
                    res = cloud_result
                self.lc_reporter.update_for_collaboration_inference()
        return [is_hard_example, edge_result]