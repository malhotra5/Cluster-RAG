import argparse
import gdown
import os
import urllib.request
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from langchain_core.prompts.chat import HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from ReRankRetriever import ReRankRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import pickle as pkl
from langchain.embeddings import HuggingFaceEmbeddings


stack_types = ["rerank", "zeroshot", "cluster", "hybrid", "context-filter"]
vector_dbs = ["llm_embed", "miniLM", "sfr_mistral", "gpt4"]
re_ranker = ["colbert", "bge"]


vector_db_names = {"miniLM": ["all-MiniLM-L6-v2DB"],
                   "llm_embed": ["llm-embedder"],
                   "sfr_mistral": ["sfr-mistralDB"],
                   "gpt4": ["gpt4allDB"]}


vector_db_links = {"miniLM": ["https://drive.google.com/drive/folders/1ovegQRv2Gp8od8Zyfy3C9MaUwq0nomen?usp=drive_link"],
                   "llm_embed": ["https://drive.google.com/drive/folders/1zLm_zd2nzEs5vDprPSx1XzEr0b5xcPk5?usp=drive_link"],
                   "gpt4": ["https://drive.google.com/drive/folders/1aU072AN8_vxFXWZlqPCsXVn9TvOZ-Jq6?usp=drive_link"]}


llama_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentence maximum and keep the answer CONCISE. Keep the answer CONCISE.\nQuestion: {question} \nContext: {context} \nAnswer:"

def download_vector_db(dbname):
    print("Downloading and preparing vector databases")
    for count, i in enumerate(vector_db_names[dbname]):
        if not os.path.isdir(i):
            gdown.download_folder(vector_db_links[dbname][count])


def create_vector_db(dbname):
    print("Preparing vectoring db")
    


def download_generation_model():
    if "llama-2-7b-chat.Q6_K.gguf" not in os.listdir():
        print("Downloading Llama2 model")
        urllib.request.urlretrieve("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf", "llama-2-7b-chat.Q6_K.gguf")
    else:
        print("Skipping download as LLama is already downloaded")



def get_retriever(retriever_type, dbname, rerank_model=None):
    
    # Load vector stores with correct embedding models
    vectorstores = []
    for i in vector_db_names[dbname]:
        if dbname == "miniLM":
            embedding_function = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


        if dbname == "llm_embed":
            model_name = "BAAI/llm-embedder"
            model_kwargs = {"device": "cuda"}
            embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
        
        
        
        store = Chroma(persist_directory=i, embedding_function=embedding_function)
        
        print(i, model_name)
        print(store.similarity_search("What is buggy?"))
        vectorstores.append(store)



    # Create retriever schemes using vectorstores
    if retriever_type == "rerank":
        # TODO: Pass appropriate rerank model here
        model = SentenceTransformer('BAAI/bge-reranker-base')
        return ReRankRetriever(vectorstore=vectorstores[0].as_retriever(), model=model)

    
    if retriever_type == "hybrid":
        retriever = vectorstores[0].as_retriever(search_kwargs={"k": 4})
        if not os.path.isfile("all_splits.pkl"):
            gdown.download("https://drive.google.com/file/d/1dx44bGE_F8o3YAW2zEvF2mf_Td75r2FF/view?usp=drive_link", "all_splits.pkl")

        print("Setting up BM25")


        with open('all_splits.pkl','rb') as f:
            all_splits = pkl.load(f)

        def flatten_extend(matrix):
            flat_list = []
            for row in matrix:
                flat_list.extend(row)
            return flat_list


        bm25_retriever = BM25Retriever.from_documents(flatten_extend(all_splits))
        return EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])



def get_chain(retriever, llm,  custom_prompt=None):
    rag_prompt = hub.pull("rlm/rag-prompt")
    # rag_prompt.messages
    if custom_prompt != None:
        prompt = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template=custom_prompt))
        rag_prompt.messages = [prompt]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )


    return qa_chain


def get_questions(filename):
    f = open(filename, "r")
    questions = f.readlines()
    f.close()

    questions = [i.strip() for i in questions]
    return questions



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(stack_types),
    )

    parser.add_argument(
        "--vector_db",
        default=None,
        type=str,
        required=True,
        help="Vector database type selected in the list: " + ", ".join(vector_dbs),
    )

    parser.add_argument(
        "--rerank_model",
        default=None,
        type=str,
        required=False,
        help="Re-ranking models type selected in the list: " + ", ".join(re_ranker),
    )

    parser.add_argument(
        "--test_set_path",
        default=None,
        type=str,
        required=True,
        help="Path for test set .txt file",
    )

    parser.add_argument(
        "--system_out_path",
        default=None,
        type=str,
        required=True,
        help="Path for storing system generated outputs",
    )


    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Whether or not to use CUDA"
    )

    args, unknown_args = parser.parse_known_args()

    model_type = args.model_type
    vector_db = args.vector_db
    rerank_model = args.rerank_model
    in_file = args.test_set_path
    out_file = args.system_out_path


    if model_type not in stack_types:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(model_type, stack_types))
    
    if vector_db not in vector_dbs:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(vector_db, vector_dbs))

    if (model_type == "hybrid" or  model_type == "rerank") and rerank_model == None:
        raise ValueError("Have chosen a hybrid or re-ranker RAG stack without without specifying on the re-ranking models: {}".format(re_ranker))
    
    if rerank_model != None and rerank_model not in re_ranker:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(rerank_model, re_ranker))



    download_vector_db(vector_db)
    download_generation_model()



    n_gpu_layers = 32 
    n_batch = 512 

    llm = LlamaCpp(
        model_path="llama-2-7b-chat.Q6_K.gguf",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=2048,
        f16_kv=True, 
        verbose=True,
    )


    retriever = get_retriever(model_type, vector_db)
    print("Running retiever test on query 'What is buggy?")
    print(retriever.get_relevant_documents("What is buggy?"))


    qa_chain = get_chain(retriever, llm,  custom_prompt=llama_prompt)
    questions = get_questions(in_file)

    from tqdm import tqdm


    f = open(out_file, "w")
    f.close()

    answers = []
    for i in tqdm(range(len(questions))):
        response = qa_chain.invoke(questions[i])

        f = open(out_file, "a")
        f.write(response + "\n")
        f.close()
        answers.append(response)







main()