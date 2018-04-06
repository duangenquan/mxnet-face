# https://github.com/cmusatyalab/openface/blob/master/Dockerfile
# https://github.com/cmusatyalab/openface/blob/master/opencv-dlib-torch.Dockerfile

FROM ubuntu:16.04

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    gcc \
    git \
    libpq-dev \
    cmake \
    curl \
    python2.7 \
    libopenblas-dev \
    liblapack-dev \
    libopencv-dev \
    python-opencv \
    libboost-all-dev \
    python-dev \
    python-setuptools \
    python-pip \
    libgfortran3 \
    ssh \
    && apt-get autoremove \
    && apt-get clean

RUN cd ~ && \
    mkdir -p dlib-tmp && \
    cd dlib-tmp && \
    curl -L \
         https://github.com/davisking/dlib/archive/v19.0.tar.gz \
         -o dlib.tar.bz2 && \
    tar xf dlib.tar.bz2 && \
    cd dlib-19.0/python_examples && \
    mkdir build && \
    cd build && \
    cmake ../../tools/python && \
    cmake --build . --config Release && \
    cp dlib.so /usr/local/lib/python2.7/dist-packages && \
    rm -rf ~/dlib-tmp

COPY . /opt/app/
RUN cd /opt/app/mxnet \ 
    make -j 4 USE_OPENCV=1 USE_BLAS=openblas \
    cd python \
    pip install --upgrade pip \
    pip install -e .


