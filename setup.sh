#!/bin/bash

# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
git clone https://github.com/stanford-futuredata/ColBERT.git
git clone https://github.com/abertsch72/unlimiformer.git
cp utils/api_run_gen.py unlimiformer/src/api_run_gen.py