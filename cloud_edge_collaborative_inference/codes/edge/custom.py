import os
from copy import deepcopy

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase

class SplitInference(JobBase):
    def __init__(self, estimator=None, hard_example_mining: dict = None):
        super(SplitInference, self).__init__(estimator=estimator)
        self.job_kind = K8sResourceKind.JOINT_INFERENCE_SERVICE.value
        self.local_ip = get_host_ip()
        self.remote_ip = self.get_parameters(
            "BIG_MODEL_IP", self.local_ip)
        self.port = int(self.get_parameters("BIG_MODEL_PORT", "5000"))

        report_msg = {
            "name": self.worker_name,
            "namespace": self.config.namespace,
            "ownerName": self.job_name,
            "ownerKind": self.job_kind,
            "kind": "inference",
            "results": []
        }
        period_interval = int(self.get_parameters("LC_PERIOD", "30"))
        self.lc_reporter = LCReporter(lc_server=self.config.lc_server,
                                      message=report_msg,
                                      period_interval=period_interval)
        self.lc_reporter.setDaemon(True)
        self.lc_reporter.start()

        if callable(self.estimator):
            self.estimator = self.estimator()
            self.estimator.load(self.config.model_url)

        self.cloud = ModelClient(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)


    def inference(self, data=None, post_process=None, **kwargs):
        res = self.estimator.predict(data, **kwargs)
        edge_result = deepcopy(res)

        self.lc_reporter.update_for_edge_inference()

        try:
            cloud_data = self.cloud.inference(
                data.tolist(), post_process=post_process, **kwargs)
            cloud_result = cloud_data["result"]
        except Exception as err:
            self.log.error(f"get cloud result error: {err}")
        else:
            res = cloud_result
        self.lc_reporter.update_for_collaboration_inference()
        return [edge_result, cloud_result]