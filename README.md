# CRAG
Cluster-RAG


## Installation

```
    conda create -n ragenv python=3.10
    pip install -r requirements.txt
```


## Creating Baselines

First prepare index as follows 

```
python baseline/prepare_index.py --data_path data/doc_chunks.pkl --vector_db gpt4all --custom_name myindex
```


