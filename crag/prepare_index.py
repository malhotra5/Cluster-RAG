import argparse

# RAG.index(
#   collection=all_documents,
#   document_metadatas=all_metadata,
#   index_name="FinalCobertIndex",
#   max_document_length=500,
#   split_documents=False
# )



index_types = ["colbert"]

def create_chunks():
    pass

def create_clusters(randomize):
    pass



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Directory where data needs to be indexed",
    )


    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Whether or not to use CUDA"
    )



    args, unknown_args = parser.parse_known_args()
    data_dir = args.data_dir





main()