from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch


class ColbertReRankRetriever(BaseRetriever):
    vectorstore : RetrieverLike

    tokenizer : AutoTokenizer

    model : AutoModel

    def maxsim(query_embedding, document_embedding):
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)
        sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)
        avg_max_sim = torch.mean(max_sim_scores, dim=1)
        return avg_max_sim


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        docs = self.vectorstore.get_relevant_documents(query, k=self.rerank_num)
 
        query_encoding = self.tokenizer(query, return_tensors='pt')
        query_embedding = self.model(**query_encoding).last_hidden_state.mean(dim=1)

        scores = []
        for document in docs:
            document_encoding = self.tokenizer(document.page_content, return_tensors='pt', truncation=True, max_length=512)
            document_embedding = self.model(**document_encoding).last_hidden_state
            score = self.maxsim(query_embedding.unsqueeze(0), document_embedding)
            scores.append((score, document))

        sorted_data = sorted(scores, key=lambda x: x[0], reverse=True)

        return [i for _,i in sorted_data][0:4]