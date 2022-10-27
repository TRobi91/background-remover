FROM nvidia/cuda:11.6.0-runtime-ubuntu18.04

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

RUN mkdir -p ~/.u2net

RUN gdown https://drive.google.com/uc?id=1rcGrMMRv5j7f6qD-fbLnZdAhsNZV6RWf -O ~/.u2net/u2netp.onnx
RUN gdown https://drive.google.com/uc?id=1grnx_D7Ql1MLfStOfWk1SccTdOb3qtpz -O ~/.u2net/u2net.onnx
RUN gdown https://drive.google.com/uc?id=1AIbrxhEuy0JXW9uDtYVyqgiCzBUU0QwU -O ~/.u2net/u2net_human_seg.onnx
RUN gdown https://drive.google.com/uc?id=1QbqEeOE9ol7SsCpro167QvkgvDVarGtc -O ~/.u2net/u2net_cloth_seg.onnx

ENV APP_HOME /rembg
WORKDIR $APP_HOME

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
#CMD exec gunicorn --bind :5000 --workers 1 --threads 8 --timeout 0 --pythonpath=/rembg/rembg --worker-class uvicorn.workers.UvicornWorker main:app
