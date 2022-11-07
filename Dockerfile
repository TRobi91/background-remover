FROM python:3.9.15 AS build

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && echo "Running $(conda --version)" && \
    conda init bash && \
    . /root/.bashrc && \
    conda update conda && \
    conda create -n python-app && \
    conda activate python-app
RUN conda install -n python-app python=3.9
RUN conda install -n python-app pip
RUN conda install -n python-app numba
RUN conda install -n python-app gcc_linux-64 gxx_linux-64
RUN conda install -n python-app scipy

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9
RUN python3.9 -m pip install numba
RUN python3.9 -m pip install scipy

RUN echo 'conda activate python-app' >> /root/.bashrc

ENV APP_HOME /rembg
WORKDIR $APP_HOME
COPY rembg/alpha_matting_compile.py rembg/alpha_matting_compile.py

RUN ["/bin/bash", "-l", "-c", "python3.9 rembg/alpha_matting_compile.py"]

FROM nvidia/cuda:11.6.0-runtime-ubuntu18.04

COPY --from=build /rembg /rembg

ENV DEBIAN_FRONTEND noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list || true
RUN rm /etc/apt/sources.list.d/nvidia-ml.list || true
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt update -y
RUN apt upgrade -y
RUN apt install -y curl software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.9 python3.9-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.9

RUN pip install gdown

RUN mkdir -p /rembg/.u2net

RUN gdown https://drive.google.com/uc?id=1rcGrMMRv5j7f6qD-fbLnZdAhsNZV6RWf -O /rembg/.u2net/u2netp.onnx
RUN gdown https://drive.google.com/uc?id=1grnx_D7Ql1MLfStOfWk1SccTdOb3qtpz -O /rembg/.u2net/u2net.onnx
RUN gdown https://drive.google.com/uc?id=1AIbrxhEuy0JXW9uDtYVyqgiCzBUU0QwU -O /rembg/.u2net/u2net_human_seg.onnx
RUN gdown https://drive.google.com/uc?id=1QbqEeOE9ol7SsCpro167QvkgvDVarGtc -O /rembg/.u2net/u2net_cloth_seg.onnx

ENV APP_HOME /rembg
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD exec gunicorn --bind 0.0.0.0:5000 --workers 4 --threads 8 --timeout 0 --pythonpath=/rembg/rembg --log-level=debug --worker-class uvicorn.workers.UvicornWorker main:app
