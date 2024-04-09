import argparse
import os
import pickle as pkl
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import shutil


vector_dbs = ["llm_embed", "miniLM", "sfr_mistral", "gpt4all"]


def createDB(persistantName, embeddingFunction, splits):
    for i in splits:
        vectorstore = Chroma.from_documents(documents=i, embedding=embeddingFunction, persist_directory=persistantName)

def create_vector_db(dbname, datapath, custom_name=None, overwrite=False, gpu=True):

    if not os.path.exists("indexes"):
        os.mkdir("indexes")

    if custom_name != None:
        index_path = "indexes/" + custom_name
        if os.path.exists(index_path):
            if not overwrite:
                raise ValueError("Index {} already exists please use --overwrite flag to overwrite".format(custom_name))
    else:
        index_path = "indexes/" + dbname
        if os.path.exists(index_path):
            if not overwrite:
                raise ValueError("Index {} already exists please use --overwrite flag to overwrite".format(dbname))
        

    if overwrite:
        try:
            shutil.rmtree(index_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    print("Loading documents")
    with open(datapath,'rb') as f: 
        all_splits = pkl.load(f)


    print("Loading embedding model")
    if dbname == "llm_embed":
        if gpu:
            model_kwargs = {"device": "cuda"}
        else:
            model_kwargs = {"device": "cpu"}
        model_name = "BAAI/llm-embedder"
        embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    elif dbname == "gpt4all":
        from langchain_community.embeddings import GPT4AllEmbeddings
        embedding_function = GPT4AllEmbeddings()

    elif dbname == "miniLM":
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
    else:
        raise ValueError("Current {} index type is not supported; please add to prepare_index.py".format(dbname))

    print("Creating db")
    createDB(index_path, embedding_function, all_splits)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=True,
        help="File path to pickle file",
    )

    parser.add_argument(
        "--vector_db",
        default=None,
        type=str,
        required=True,
        help="Vector database type selected in the list: " + ", ".join(vector_dbs),
    )

    parser.add_argument(
        "--custom_name",
        default=None,
        type=str,
        required=False,
        help="Custom name to store the index",
    )


    parser.add_argument(
        "--overwrite",
        action="store_false",
        help="Whether or not to overwrite existing index"
    )

    parser.add_argument(
            "--gpu",
            action="store_true",
            help="Whether or not to use CUDA"
    )




    args, unknown_args = parser.parse_known_args()
    vector_db = args.vector_db
    data_path = args.data_path
    custom_name = args.custom_name
    overwrite = args.overwrite
    gpu = args.gpu
    

    if vector_db not in vector_dbs:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(vector_db, vector_dbs))
    
    if not os.path.isfile(data_path):
        raise ValueError("Invalid path to documents detected: {}".format(data_path))


    create_vector_db(vector_db, data_path, custom_name=custom_name, overwrite=overwrite, gpu=gpu)

    print("Finished creating dataset")


main()