FROM nvidia/cuda:9.0-cudnn7-devel
MAINTAINER Haotian zhang <haotianz@stanford.edu>

RUN apt-get update && apt-get install -y rsync openssh-server vim wget unzip htop tmux 
RUN apt-get install -y libsm6 libxrender1 libgtk2.0-dev

RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip        

RUN pip3 install torch>=1.0.0 torchvision
RUN pip3 install ipykernel jupyter 
RUN pip3 install opencv-contrib-python matplotlib 
RUN pip3 install cupy scipy

EXPOSE 8888