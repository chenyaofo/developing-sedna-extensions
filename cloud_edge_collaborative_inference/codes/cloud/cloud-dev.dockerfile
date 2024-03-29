FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    TZ=Asia/Shanghai \
    TERM=xterm-256color \
    PATH=/opt/conda/bin:$PATH \
    PYTHON_VERSION=3.10


RUN APT_INSTALL="apt-get install -y --no-install-recommends --no-install-suggests" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL tzdata && \
    ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install wget -y

RUN wget https://mirror.ghproxy.com/https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O ~/installer.sh && \
    /bin/bash ~/installer.sh -b -p /opt/conda && \
    rm ~/installer.sh && \
    /opt/conda/bin/conda init bash && \
    echo "conda activate dev" >> $HOME/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete


RUN /opt/conda/bin/mamba create -n dev python=${PYTHON_VERSION} && \
    CONDA_INSTALL="/opt/conda/bin/mamba install -n dev -y" && \
    /opt/conda/bin/mamba clean -afy && \
    /opt/conda/bin/mamba install git && \
    echo 'Conda Install Done!' 

COPY requirements.txt /workspace/
WORKDIR /workspace

RUN PIP_INSTALL="/opt/conda/envs/dev/bin/pip install --no-cache-dir" && \
    $PIP_INSTALL -r /workspace/requirements.txt -i https://pypi.scut-smil.cn/simple

COPY . /workspace/
RUN mv /workspace/inference.py ls /opt/conda/envs/dev/lib/python3.10/site-packages/sedna/service/server/inference.py
# ENTRYPOINT ["nvidia-smi"]
# CMD [ "tail", "-f", "/dev/null" ]
RUN mkdir tmp/imagenet1000 -p
ENTRYPOINT ["/opt/conda/envs/dev/bin/python"]
CMD ["main.py"]  
