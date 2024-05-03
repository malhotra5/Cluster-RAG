from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)        
# from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import sys
sys.path.insert(0, '../ColBERT/')
from colbert.infra import Run, RunConfig
from colbert import Searcher
import pickle as pkl
import os

vector_dbs = ["llm_embed", "miniLM", "sfr_mistral", "gpt4all", "colbert"]


def get_llm_embed(gpu):
    if gpu:
        model_kwargs = {"device": "cuda"}
    else:
        model_kwargs = {"device": "cpu"}
    model_name = "BAAI/llm-embedder"
    embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    return embedding_function


def get_gpt4all():
    embedding_function = GPT4AllEmbeddings()
    return embedding_function

def get_minilm():
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def get_colbert():
    return None


def get_bge_rerank():
    # rerank_model = SentenceTransformer('BAAI/bge-reranker-base')
    rerank_model = pipeline("text-classification", model="BAAI/bge-reranker-base")
    return rerank_model


def get_colbert_rerank():
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    return [tokenizer, model]


def get_model(name, gpu=False):
    if name == "llm_embed":
        embedding_function = get_llm_embed(gpu)

    elif name == "gpt4all":
        embedding_function = get_gpt4all()

    elif name == "miniLM":
        embedding_function = get_minilm()

    elif name == "colbert":
        embedding_function = get_colbert()
    else:
        raise ValueError("Current {} model type is not supported; please add to supported_models.py".format(name))

    return embedding_function


def get_rerank_model(name):
    if name == "bge":
        model = get_bge_rerank()
    
    elif name == "colbert":
        model = get_colbert_rerank() 

    else:
        raise ValueError("Current {} model is not supported for reranking; please add to supported_models.py".format(name))

    return model


def get_cluster_model(name):
    indexes = []
    paths = os.listdir("indexes/" + name)
    for i in paths:
        
        with open("indexes/{}/{}/data.pkl".format(name, i), 'rb') as f:
            data = pkl.load(f)

        with Run().context(RunConfig(experiment='notebook')):
            searcher = Searcher(index="{}/indexes/{}/{}".format(os.getcwd(),name, i), collection=data)
            indexes.append(searcher)

    return indexes

