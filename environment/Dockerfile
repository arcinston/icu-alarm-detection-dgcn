# hash:sha256:2dd6b28d5b6e359baedba5fd93eb65d1f701e0719c5be086b4b722ba05c97025
FROM registry.codeocean.com/codeocean/miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.3.4 \
        pandas==1.2.3 \
        pytorch==1.4.0 \
        scikit-learn==0.24.1 \
        scipy==1.6.1 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    prettytable==2.1.0 \
    tensorboardx==2.1 \
    tensorwatch==0.9.1 \
    torchsummary==1.5.1 \
    wfdb==3.3.0
