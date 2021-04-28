#! /bin/bash
#
# file: setup.sh
#
#
# This bash script performs any setup necessary in order to test your entry.
# It is run only once, before running any other code belonging to your entry.
echo "=== Run Entry Setup Script ==="
#Remove (or set it to 0) if you are not using MATLAB
NEED_MATLAB=0

#Define the event type that you are competing on. 
#Use the following coding:
#      1 -Real-time only
#      2 -Retrospective only
#      3 -Both

if [ ! -d "./miniconda3" ];then
  chmod 777 Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda3
  rm Miniconda3-latest-Linux-x86_64.sh
else
  echo "miniconda3 have installed"
fi
chmod -R 777 ./miniconda3/bin/

./miniconda3/bin/conda --version 
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/certifi-2020.4.5.1-py37_0.conda
rm ./conda_pkgs/certifi-2020.4.5.1-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/chardet-3.0.4-py37_1003.conda
rm ./conda_pkgs/chardet-3.0.4-py37_1003.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/cycler-0.10.0-py37_0.conda
rm ./conda_pkgs/cycler-0.10.0-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/freetype-2.9.1-h8a8886c_1.conda
rm ./conda_pkgs/freetype-2.9.1-h8a8886c_1.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/intel-openmp-2019.4-243.conda
rm ./conda_pkgs/intel-openmp-2019.4-243.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/joblib-0.13.2-py37_0.conda
rm ./conda_pkgs/joblib-0.13.2-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/mkl-2020.1-217.conda
rm ./conda_pkgs/mkl-2020.1-217.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/mkl-service-2.3.0-py37he904b0f_0.conda
rm ./conda_pkgs/mkl-service-2.3.0-py37he904b0f_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/mkl_random-1.1.1-py37h0573a6f_0.conda
rm ./conda_pkgs/mkl_random-1.1.1-py37h0573a6f_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/mkl_fft-1.0.15-py37ha843d7b_0.conda
rm ./conda_pkgs/mkl_fft-1.0.15-py37ha843d7b_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/jpeg-9b-habf39ab_1.conda
rm ./conda_pkgs/jpeg-9b-habf39ab_1.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/libtiff-4.1.0-h2733197_0.conda
rm ./conda_pkgs/libtiff-4.1.0-h2733197_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/libgfortran-ng-7.3.0-hdf63c60_0.conda
rm ./conda_pkgs/libgfortran-ng-7.3.0-hdf63c60_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/libpng-1.6.37-hbc83047_0.conda
rm ./conda_pkgs/libpng-1.6.37-hbc83047_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/blas-1.0-mkl.conda
rm ./conda_pkgs/blas-1.0-mkl.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/openssl-1.1.1g-h7b6447c_0.conda
rm ./conda_pkgs/openssl-1.1.1g-h7b6447c_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/idna-2.8-py37_0.conda
rm ./conda_pkgs/idna-2.8-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/idna_ssl-1.1.0-py37_0.conda
rm ./conda_pkgs/idna_ssl-1.1.0-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/numpy-base-1.18.1-py37hde5b4d6_1.conda
rm ./conda_pkgs/numpy-base-1.18.1-py37hde5b4d6_1.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/numpy-1.18.1-py37h4f9e942_0.conda
rm ./conda_pkgs/numpy-1.18.1-py37h4f9e942_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pip-20.0.2-py37_3.conda
rm ./conda_pkgs/pip-20.0.2-py37_3.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/matplotlib-3.1.3-py37_0.conda
rm ./conda_pkgs/matplotlib-3.1.3-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/matplotlib-base-3.1.3-py37hef1b27d_0.conda
rm ./conda_pkgs/matplotlib-base-3.1.3-py37hef1b27d_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pandas-1.0.3-py37h0573a6f_0.conda
rm ./conda_pkgs/pandas-1.0.3-py37h0573a6f_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/zstd-1.3.7-h0b5b093_0.conda
rm ./conda_pkgs/zstd-1.3.7-h0b5b093_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/scipy-1.4.1-py37habc2bb6_0.conda
rm ./conda_pkgs/scipy-1.4.1-py37habc2bb6_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pillow-7.1.2-py37hb39fc2d_0.conda
rm ./conda_pkgs/pillow-7.1.2-py37hb39fc2d_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/olefile-0.46-py37_0.conda
rm ./conda_pkgs/olefile-0.46-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/nose-1.3.7-py37_2.conda
rm ./conda_pkgs/nose-1.3.7-py37_2.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/kiwisolver-1.2.0-py37hfd86e86_0.conda
rm ./conda_pkgs/kiwisolver-1.2.0-py37hfd86e86_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pyparsing-2.3.1-py37_0.conda
rm ./conda_pkgs/pyparsing-2.3.1-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/python-dateutil-2.8.0-py37_0.conda
rm ./conda_pkgs/python-dateutil-2.8.0-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pytz-2018.9-py37_0.conda
rm ./conda_pkgs/pytz-2018.9-py37_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/scikit-learn-0.22.1-py37hd81dba3_0.conda
rm ./conda_pkgs/scikit-learn-0.22.1-py37hd81dba3_0.conda
./miniconda3/bin/conda install --use-local -y ./conda_pkgs/pytorch-1.4.0-cpu_py37h7e40bad_0.conda
rm ./conda_pkgs/pytorch-1.4.0-cpu_py37h7e40bad_0.conda
./miniconda3/bin/pip install ./pip_pkgs/threadpoolctl-2.1.0-py3-none-any.whl
rm ./pip_pkgs/threadpoolctl-2.1.0-py3-none-any.whl
./miniconda3/bin/pip install ./pip_pkgs/sklearn-0.0.tar.gz
rm ./pip_pkgs/sklearn-0.0.tar.gz
./miniconda3/bin/pip install ./pip_pkgs/wfdb-2.2.1.tar.gz
rm ./pip_pkgs/wfdb-2.2.1.tar.gz
./miniconda3/bin/conda list
./miniconda3/bin/python --version
# 删除软链接，建立新的软连接

EVENT_TYPE=1
chmod a+x ./challenge.py
echo "=== Entry Setup Script was run successfully ==="
