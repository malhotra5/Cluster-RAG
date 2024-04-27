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
from CRAGRetriever import CRAGRetriever
import supported_models
from tqdm import tqdm
import transformers
from transformers import LogitsProcessor
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
import sys
sys.path.insert(0, '../unlimiformer/src')
from run_generation import unlimiform, generate
# from colbert import Searcher



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


llama_prompt = "<s>[INST] <<SYS>>\nYou are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use one sentence maximum and keep the answer CONCISE. Keep the answer CONCISE.\n<</SYS>>\n\nQuestion: {question} \nContext: {context} \nAnswer: [/INST]"

# def download_vector_db(dbname):
#     print("Downloading and preparing vector databases")
#     for count, i in enumerate(vector_db_names[dbname]):
#         if not os.path.isdir(i):
#             gdown.download_folder(vector_db_links[dbname][count])


# class EosTokenRewardLogitsProcessor(LogitsProcessor):
#   def __init__(self,  eos_token_id: int, max_length: int):
    
#         if not isinstance(eos_token_id, int) or eos_token_id < 0:
#             raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

#         if not isinstance(max_length, int) or max_length < 1:
#           raise ValueError(f"`max_length` has to be a integer bigger than 1, but is {max_length}")

#         self.eos_token_id = eos_token_id
#         self.max_length=max_length

#   def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#     cur_len = input_ids.shape[-1]
#     # start to increese the reward of the  eos_tokekn from 80% max length  progressively on length
#     for cur_len in (max(0,int(self.max_length*0.8)), self.max_length ):
#       ratio = cur_len/self.max_length
#       num_tokens = scores.shape[1] # size of vocab
#       scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]] =\
#       scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]*ratio*10*torch.exp(-torch.sign(scores[:, [i for i in range(num_tokens) if i != self.eos_token_id]]))
#       scores[:, self.eos_token_id] = 1e2*ratio
#     return scores



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

    if retriever_type != "cluster":
        store = Chroma(persist_directory="indexes/" + dbname, embedding_function=embedding_function)
    else:
        indexes = supported_models.get_cluster_model(dbname)

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

    if retriever_type == "cluster":
        return CRAGRetriever(vectorstore=indexes)

    raise ValueError("Current {} stack type is not supported; please add".format(retriever_type))



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

    parser.add_argument(
        "--use_unlimiform",
        action="store_true",
        help="Use for cluster when holding more than 2 clusters"
    )

    args, unknown_args = parser.parse_known_args()

    model_type = args.model_type
    vector_db = args.vector_db
    rerank_model = args.rerank_model
    in_file = "questions/" + args.questions_file
    out_file = args.system_out_path
    use_unlimiform = args.use_unlimiform
    


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
    # download_generation_model()



    # n_gpu_layers = 32 
    # n_batch = 512 

    # llm = LlamaCpp(
    #     model_path="llama-2-7b-chat.Q6_K.gguf",
    #     n_gpu_layers=n_gpu_layers,
    #     n_batch=n_batch,
    #     n_ctx=2048,
    #     f16_kv=True, 
    #     verbose=True,
    # )


    if not use_unlimiform:
        model_dir = "meta-llama/Llama-2-7b-chat-hf"
        hf_auth = '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'
        model_config = transformers.AutoConfig.from_pretrained(
            model_dir,
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            config=model_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
        model.eval()
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            use_auth_token=hf_auth
        )

        generate_text = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            # we pass model parameters here too
            # stopping_criteria=stopping_criteria,  # without this model rambles during chat
            temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=512,  # max number of tokens to generate in the output
            repetition_penalty=1.1  # without this output begins repeating
        )

        llm = HuggingFacePipeline(pipeline=generate_text)
    else:
        model, tokenizer = unlimiform()


        llm = transformers.pipeline(
            model=model, 
            tokenizer=tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.1,
            max_new_tokens=512,
            repetition_penalty=1.1
        )




    retriever = get_retriever(model_type, vector_db, llm)
    print("Running retiever test on query 'What is buggy?")
    print(retriever.get_relevant_documents("What is buggy?"))


    qa_chain = get_chain(retriever, generate(model, tokenizer),  custom_prompt=llama_prompt)
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