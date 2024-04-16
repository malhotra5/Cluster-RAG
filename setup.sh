#!/bin/bash

CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
pip uninstall faiss-cpu -y
conda install -c conda-forge faiss-gpu -y
export CUDA_HOME=$CONDA_PREFIX
conda install -c nvidia cuda-toolkit -y