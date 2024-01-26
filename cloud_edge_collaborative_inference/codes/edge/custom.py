import os
from copy import deepcopy
import time
import numpy as np
import sys

from sedna.common.utils import get_host_ip
from sedna.common.class_factory import ClassFactory, ClassType
from sedna.service.server import InferenceServer
from sedna.service.client import ModelClient, LCReporter,http_request
from sedna.common.constant import K8sResourceKind
from sedna.core.base import JobBase

class ModelClient_time(ModelClient):
    def inference(self, x, **kwargs):
        """Use the remote big model server to inference."""
        json_data = deepcopy(kwargs)
        json_data.update({"data": x})
        json_data.update({'uploadtime':time.time()})
        _url = f"{self.endpoint}/predict"
        return http_request(url=_url, method="POST", json=json_data)

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

        # if callable(self.estimator):
        #     self.estimator = self.estimator()
        self.estimator.load(self.config.model_url)

        self.cloud = ModelClient_time(service_name=self.job_name,
                                 host=self.remote_ip, port=self.port)


    def inference(self, data=None, post_process=None, **kwargs):
        edge_time_start = time.time()
        res = self.estimator.predict(data, **kwargs)
        edge_time_end = time.time()
        edge_use_time = edge_time_end-edge_time_start
        edge_result = deepcopy(res)
        print(sys.getsizeof(np.array(res)))
        print(np.array(res).shape)
        self.lc_reporter.update_for_edge_inference()
        try:
            res = self.cloud.inference(
                res, post_process=post_process, **kwargs)
            cloud_result,upload_time,cloud_use_time, cloud_return_time= \
                res['result'],res['uploadusetime'],res['cloudusetime'],res['cloudreturntime']
        except Exception as err:
            self.log.error(f"get cloud result error: {err}")
        else:
            res = cloud_result
        cloud_down_time = time.time()-cloud_return_time
        self.lc_reporter.update_for_collaboration_inference()
        return [edge_result, cloud_result,cloud_use_time, edge_use_time,upload_time,cloud_down_time]
