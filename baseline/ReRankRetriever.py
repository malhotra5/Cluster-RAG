from langchain_core.retrievers import BaseRetriever, RetrieverLike
# from langchain_core.callbacks import CallbackManagerForRetrieverRun
# from langchain_core.documents import Document
# from typing import List
# from sentence_transformers import SentenceTransformer


class ReRankRetriever(BaseRetriever):
    # raise ValueError("Please make sure to upgrade transformers and sentence-transformers library before using rerank retriever")
    # vectorstore : RetrieverLike

    # model : SentenceTransformer

    # rerank_num : int

    # def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

    #     docs = self.vectorstore.get_relevant_documents(query, k=self.rerank_num)
 
    #     queries = [query]
    #     sentences = []
    #     for i in docs:
    #         sentences.append(i.page_content)

    #     embeddings_1 = self.model.encode(sentences, normalize_embeddings=True)
    #     embeddings_2 = self.model.encode(queries, normalize_embeddings=True)
    #     similarity = embeddings_1 @ embeddings_2.T
    #     results = [(similarity[count][0], i) for count, i in enumerate(docs)]
    #     results = sorted(results, key=lambda x:x[0])

    #     return [i for _,i in results][0:4]

    pass