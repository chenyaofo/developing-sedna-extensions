import os
from sedna.service.server import InferenceServer
from sedna.core.joint_inference import BigModelService


def start(self: BigModelService):
    """
    Start inference rest server
    """
    if self.config.model_url.startswith("s3:"):
        self.estimator.load(self.config.model_url)
    else:
        raise FileExistsError(f"{self.config.model_url} is wrong")
    app_server = InferenceServer(model=self, servername=self.job_name,
                                 host=self.local_ip, http_port=self.port)
    app_server.start()


def sedna_patch():
    BigModelService.start = start
