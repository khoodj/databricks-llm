import os
import base64
import yaml
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from io import BytesIO
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from operator import itemgetter
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from ..utils.agents import SQLAgent
from ..prompts.prompt import *
from ..utils.lambda_functions import LambdaFunctions

# loading environment variables
load_dotenv()


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


class SQLChain(SQLAgent):
    def reformulate_question_chain(self):

        reformulate_question_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )
            | self.get_llm()
            | StrOutputParser()
        )
        return reformulate_question_chain

    def output_chain(self):
        agent_chain = {
            "input": self.reformulate_question_chain(),
        } | self.get_agent_executor()
        # agent_chain = self.get_agent_executor()

        sql_chain_w_history = agent_chain | RunnableLambda(LambdaFunctions.strip_path)

        final_chain = RunnableWithMessageHistory(
            sql_chain_w_history,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
            output_messages_key="output",
        )

        return final_chain
