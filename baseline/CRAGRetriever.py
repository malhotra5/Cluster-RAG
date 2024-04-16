from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List



class CRAGRetriever(BaseRetriever):

    vectorstore : List[RetrieverLike]

    def flatten_extend(self, matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        all_docs = []
        for i in self.vectorstore:
            all_docs.append(i.get_relevant_documents(query, k=3))

        all_docs = self.flatten_extend(all_docs)
        return all_docs