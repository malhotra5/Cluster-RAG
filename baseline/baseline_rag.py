import argparse
import gdown
import os
import urllib.request
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import HumanMessagePromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from ReRankRetriever import ReRankRetriever
from ColbertReRankRetriever import ColbertReRankRetriever
from HydeRetriever import HydeRetriever

# from langchain.embeddings import HuggingFaceEmbeddings
import supported_models
from tqdm import tqdm




stack_types = ["rerank", "naive", "cluster", "hyde", "context-filter"]
# vector_dbs = ["llm_embed", "miniLM", "sfr_mistral", "gpt4"]
vector_dbs = supported_models.vector_dbs
re_ranker = ["colbert", "bge"]


vector_db_names = {"miniLM": ["all-MiniLM-L6-v2DB"],
                   "llm_embed": ["llm-embedder"],
                   "sfr_mistral": ["sfr-mistralDB"],
                   "gpt4": ["gpt4allDB"]}


vector_db_links = {"miniLM": ["https://drive.google.com/drive/folders/1ovegQRv2Gp8od8Zyfy3C9MaUwq0nomen?usp=drive_link"],
                   "llm_embed": ["https://drive.google.com/drive/folders/1zLm_zd2nzEs5vDprPSx1XzEr0b5xcPk5?usp=drive_link"],
                   "gpt4": ["https://drive.google.com/drive/folders/1aU072AN8_vxFXWZlqPCsXVn9TvOZ-Jq6?usp=drive_link"]}


llama_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentence maximum and keep the answer CONCISE. Keep the answer CONCISE.\nQuestion: {question} \nContext: {context} \nAnswer:"

# def download_vector_db(dbname):
#     print("Downloading and preparing vector databases")
#     for count, i in enumerate(vector_db_names[dbname]):
#         if not os.path.isdir(i):
#             gdown.download_folder(vector_db_links[dbname][count])



def uniqify(filename):
    count = 0
    pathname = "generations/" + filename + "_{}.txt" 
    while os.path.isfile(pathname.format(count)):
        count += 1
    return pathname.format(count)


def download_generation_model():
    if "llama-2-7b-chat.Q6_K.gguf" not in os.listdir():
        print("Downloading Llama2 model")
        urllib.request.urlretrieve("https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf", "llama-2-7b-chat.Q6_K.gguf")
    else:
        print("Skipping download as LLama is already downloaded")



def get_retriever(retriever_type, dbname, llm, rerank_model="bge"):
    
    # Load vector stores with correct embedding models
    
    embedding_function = supported_models.get_model(dbname, True)
    store = Chroma(persist_directory="indexes/" + dbname, embedding_function=embedding_function)

    if retriever_type == "naive":
        return store.as_retriever()
    
    if retriever_type == "hyde":
        return HydeRetriever(vectorstore=store.as_retriever(), llm=llm)

    if retriever_type == "rerank":
        # TODO: Pass appropriate rerank model here
        if rerank_model == "bge":
            rerank_model = supported_models.get_rerank_model(rerank_model)
            return ReRankRetriever(vectorstore=store.as_retriever(), model=rerank_model, rerank_num=40)
        
        if rerank_model == "colbert":
            rerank_model = supported_models.get_rerank_model(rerank_model)
            return ColbertReRankRetriever(vectorstore=store.as_retriever(), tokenizer=rerank_model[0], model=rerank_model[1], rerank_num=40)


    raise ValueError("Current {} stack type is not supported; please add".format(retriever_type))

    
    # if retriever_type == "hybrid":
    #     retriever = vectorstores[0].as_retriever(search_kwargs={"k": 4})
    #     if not os.path.isfile("all_splits.pkl"):
    #         gdown.download("https://drive.google.com/file/d/1dx44bGE_F8o3YAW2zEvF2mf_Td75r2FF/view?usp=drive_link", "all_splits.pkl")

    #     print("Setting up BM25")


    #     with open('all_splits.pkl','rb') as f:
    #         all_splits = pkl.load(f)

    #     def flatten_extend(matrix):
    #         flat_list = []
    #         for row in matrix:
    #             flat_list.extend(row)
    #         return flat_list


    #     bm25_retriever = BM25Retriever.from_documents(flatten_extend(all_splits))
    #     return EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])



def get_chain(retriever, llm,  custom_prompt=None):
    rag_prompt = hub.pull("rlm/rag-prompt")
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
        "--questions_file",
        default=None,
        type=str,
        required=True,
        help="Path for test set .txt file",
    )

    parser.add_argument(
        "--system_out_path",
        default=None,
        type=str,
        required=False,
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
    in_file = "questions/" + args.questions_file
    out_file = args.system_out_path
    


    if model_type not in stack_types:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(model_type, stack_types))
    
    if vector_db not in vector_dbs:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(vector_db, vector_dbs))

    if (model_type == "hybrid" or  model_type == "rerank") and rerank_model == None:
        raise ValueError("Have chosen a hybrid or re-ranker RAG stack without without specifying on the re-ranking models: {}".format(re_ranker))
    
    if rerank_model != None and rerank_model not in re_ranker:
        raise ValueError("Unknown arguments detected: {}. Expected one of {}".format(rerank_model, re_ranker))


    if out_file == None:
        out_file = args.questions_file

    out_file = out_file.split(".txt")[0] + "_{}".format(model_type)
    if rerank_model != None:
        out_file = out_file + "_{}".format(rerank_model)




    # download_vector_db(vector_db)
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

    # model_dir = "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf"
    # device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    # bnb_config = transformers.BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # begin initializing HF items, you need an access token
    # hf_auth = '<add your access token here>'
    # model_config = transformers.AutoConfig.from_pretrained(
    #     model_dir,
    #     # use_auth_token=hf_auth
    # )

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_dir,
    #     trust_remote_code=True,
    #     config=model_config,
    #     # quantization_config=bnb_config,
    #     device_map='auto',
    #     use_auth_token=hf_auth
    # )
    # model.eval()
    
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_dir,
    #     use_auth_token=hf_auth
    # )

    # generate_text = transformers.pipeline(
    #     model=model, 
    #     tokenizer=tokenizer,
    #     return_full_text=True,  # langchain expects the full text
    #     task='text-generation',
    #     # we pass model parameters here too
    #     # stopping_criteria=stopping_criteria,  # without this model rambles during chat
    #     temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    #     max_new_tokens=512,  # max number of tokens to generate in the output
    #     repetition_penalty=1.1  # without this output begins repeating
    # )

    # llm = HuggingFacePipeline(pipeline=generate_text)





    retriever = get_retriever(model_type, vector_db, llm)
    print("Running retiever test on query 'What is buggy?")
    print(retriever.get_relevant_documents("What is buggy?"))


    qa_chain = get_chain(retriever, llm,  custom_prompt=llama_prompt)
    questions = get_questions(in_file)

    

    
    
    out_file = uniqify(out_file)
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