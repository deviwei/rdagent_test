FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# 更换源并安装系统依赖
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y \
        curl \
        vim \
        git \
        python3-dev \
        build-essential \
        libhdf5-dev \
        gcc \
        g++ \
        make \
        cmake \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/* 

# 更新 pip 并安装基础依赖
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --upgrade cython numpy pandas scikit-learn

# 克隆特定版本的 qlib
RUN git clone --depth=100 https://github.com/microsoft/qlib.git
WORKDIR /workspace/qlib
RUN git reset 4d621bff99bb403231e26bb1dddae6e77d2af170 --hard

# 安装依赖包（指定版本）
RUN pip install \
    catboost==1.2.2 \
    xgboost==2.0.3 \
    scipy==1.11.4 \
    tables==3.9.2 \
    lightgbm==4.3.0

# 安装 qlib
RUN python -m pip install .