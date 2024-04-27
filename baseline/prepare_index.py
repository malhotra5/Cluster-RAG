import argparse
import os
import pickle as pkl
from langchain_community.vectorstores import Chroma
import supported_models
import shutil
import sys
sys.path.insert(0, '../ColBERT/')
from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig
import random
import pickle as pkl

vector_dbs = supported_models.vector_dbs

def flatten_concatenation(matrix):
    flat_list = []
    for row in matrix:
        flat_list += row
    return flat_list

def group_data(splits, cluster):
    data_splits = []
    if cluster["cluster_groups"] == None:
        cluster_num = cluster["cluster_num"]
        [data_splits.append([]) for _ in range(cluster_num)]
        
        all_docs = flatten_concatenation(splits)
        random.shuffle(all_docs)
        for i in range(cluster_num):
            s = int(i/cluster_num * len(all_docs))
            e = int((i+1)/cluster_num * len(all_docs))
            data_splits[i].extend(all_docs[s:e])

    else:
        raise ValueError("Haven't implemented custom grouping yet")
    

    return data_splits



def createDB(persistantName, embeddingFunction, splits, cluster, group=None):
    if not cluster["cluster"]:
        for i in splits:
            vectorstore = Chroma.from_documents(documents=i, embedding=embeddingFunction, persist_directory=persistantName)

    else:
        checkpoint = 'colbert-ir/colbertv2.0'
        data = group_data(splits, cluster)
        for count, i in enumerate(data):
            all_documents = []
            for j in i:
                all_documents.append(j.page_content)

            with Run().context(RunConfig(nranks=1, experiment='notebook')):
                config = ColBERTConfig(doc_maxlen=500, nbits=4, kmeans_niters=4) 
                
                cluster_name = "{}/{}/cluster_{}/".format(os.getcwd(), persistantName, count)                                                           
                indexer = Indexer(checkpoint=checkpoint, config=config)
                indexer.index(name=cluster_name, collection=all_documents, overwrite=True)
            with open(cluster_name + "data.pkl", 'wb') as f:
                pkl.dump(all_documents, f)





def create_vector_db(dbname, datapath, cluster, custom_name=None, overwrite=False, gpu=True):

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
    createDB(index_path, embedding_function, all_splits, cluster=cluster)




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

    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Whether or not to create Cluster-RAG index"
    )

    parser.add_argument(
        "--num_clusters",
        default=2,
        type=int,
        required=False,
        help="Number of clusters to create"
    )

    parser.add_argument(
        "--groups",
        default=None,
        type=list,
        required=False,
        help="Grouping for clusters"
    )




    args, unknown_args = parser.parse_known_args()
    vector_db = args.vector_db
    data_path = args.data_path
    custom_name = args.custom_name
    overwrite = args.overwrite
    cluster = args.cluster
    cluster_groups = args.groups
    cluster_num = args.num_clusters
    gpu = args.gpu
    

    if vector_db not in vector_dbs:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(vector_db, vector_dbs))
    
    if not os.path.isfile(data_path):
        raise ValueError("Invalid path to documents detected: {}".format(data_path))


    create_vector_db(vector_db, 
                     data_path, 
                     cluster={"cluster": cluster,
                              "cluster_groups": cluster_groups,
                              "cluster_num": cluster_num}, 
                     custom_name=custom_name, 
                     overwrite=overwrite,
                     gpu=gpu)

    print("Finished creating dataset")


if __name__ == "__main__":
    main()