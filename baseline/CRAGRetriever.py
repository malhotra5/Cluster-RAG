from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
import sys
sys.path.insert(0, '../ColBERT/')
from colbert import Searcher
from langchain.docstore.document import Document




class CRAGRetriever(BaseRetriever):

    vectorstore : List[Searcher]

    def flatten_extend(self, matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        all_docs = []
        for i in self.vectorstore:
            res = i.search(query, k=2)
            for passage_id, _, _ in zip(*res):
                all_docs.append(i.collection[passage_id])


        # all_docs = self.flatten_extend(all_docs)
        all_docs = [Document(page_content=i) for i in all_docs]
        return all_docs