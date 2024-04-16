from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class HydeRetriever(BaseRetriever):
    vectorstore : RetrieverLike

    llm : LlamaCpp
    
    def generate_hypothetical_documents(self, question):
        template = """Please write hypothetical answers for the question.
                    Question: {question}
                    Passage:"""
        
        prompt_hyde = ChatPromptTemplate.from_template(template)
        generate_docs_for_retrieval = (
            prompt_hyde | self.llm | StrOutputParser() 
        )
        

        return generate_docs_for_retrieval.invoke({"question":question})


    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:

        hypothetical_doc = self.generate_hypothetical_documents(query)
    
        docs = self.vectorstore.get_relevant_documents(hypothetical_doc, k=4)
        return docs