from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import sys
import os
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.llm import PDF_LLM
from src.prompts.prompt import *


with open("./conf/config.yaml", "r") as file:
    config = yaml.safe_load(file)


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class RAGChain(PDF_LLM):
    def output_chain(self):
        vectorstore = FAISS.load_local(
            config["pdf_vectorstore_path"],
            embeddings=self.get_embeddings_model(),
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(k=5)

        # creating history aware retriever
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_INSTRUCTION_PROMPT),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.get_llm(), qa_prompt)

        # rag_chain = RunnablePassthrough.assign(
        #     input=contextualize_q_prompt | self.get_llm() | StrOutputParser()
        # ) | create_retrieval_chain(retriever, question_answer_chain)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # creating qa chain
        # question_answer_chain = create_stuff_documents_chain(self.get_llm(), qa_prompt)

        # retrieval chain contains at the very least a "context" and "answer" key
        # rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        final_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="answer",
        )

        return final_chain
