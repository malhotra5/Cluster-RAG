import argparse
import os
import pickle as pkl
from langchain_community.vectorstores import Chroma
import supported_models
import shutil


vector_dbs = supported_models.vector_dbs


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
    embedding_function = supported_models.get_model(dbname, gpu)
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
        action="store_true",
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