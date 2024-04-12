# CRAG
Cluster-RAG


## Installation

```
    conda create -n ragenv python=3.10
    pip install -r requirements.txt
    chmod +x ./setup.sh
    ./setup.sh
```


## Creating Baselines

First prepare index as follows 

```
python baseline/prepare_index.py --data_path data/doc_chunks.pkl --vector_db gpt4all --custom_name myindex
```

Run generations on index as follows 

```
python baseline/baseline_rag.py --vector_db miniLM --model_type naive --questions_file questions_1.txt
```


