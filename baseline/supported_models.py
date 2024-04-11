from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)        


vector_dbs = ["llm_embed", "miniLM", "sfr_mistral", "gpt4all"]


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



def get_model(name, gpu=False):
    if name == "llm_embed":
        embedding_function = get_llm_embed(gpu)

    elif name == "gpt4all":
        embedding_function = get_gpt4all()

    elif name == "miniLM":
        embedding_function = get_minilm()
        
    else:
        raise ValueError("Current {} model type is not supported; please add to supported_models.py".format(name))

    return embedding_function