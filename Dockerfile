FROM ubuntu:16.04

# Update source
#RUN sed -i 's/http:\/\/archive\.ubuntu\.com\/ubuntu\//http:\/\/mirrors\.163\.com\/ubuntu\//g' /etc/apt/sources.list
RUN apt-get update -y

# Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip

# Resolve problem of installing pillow
RUN apt-get install -y libjpeg8-dev

RUN apt-get install -y python-pip
RUN pip install --upgrade pip

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

RUN pip --no-cache-dir install -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com \
        ipython==5.3.0 \
        ipykernel==4.6.1 \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        && \
    python -m ipykernel.kernelspec

RUN apt-get install -y vim

# Install MiniFlow
RUN pip install miniflow

CMD ["bash"]
