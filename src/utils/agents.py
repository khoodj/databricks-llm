from .custom_tools import SQLCustomTools, SQLFilterUniqueValues, RAGSQLTools
from .llm import SQL_LLM
from ..prompts.prompt import *
from langchain.agents import (
    AgentExecutor,
    create_tool_calling_agent,
    create_openai_tools_agent,
)
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import AzureChatOpenAI
from logging import getLogger
from abc import abstractmethod

logger = getLogger(__name__)


class Agent(SQL_LLM):
    @classmethod
    @abstractmethod
    def get_agent_executor(cls):
        pass

    @classmethod
    def _create_agent_executor(cls, tools, agent) -> AgentExecutor:
        """Helper method to create the agent executor with given tools."""
        prompt = cls._create_prompt()

        __agent = agent(cls.get_llm(), tools, prompt)

        agent_executor = AgentExecutor(
            agent=__agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True,
        )
        return agent_executor

    @staticmethod
    def _create_prompt() -> ChatPromptTemplate:
        """Helper method to create a chat prompt."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", GENERIC_SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )


class SQLAgent(Agent):
    @classmethod
    def get_agent_executor(cls) -> AgentExecutor:
        tools = [
            SQLCustomTools.create_sql_command,
            SQLCustomTools.execute_sql_command,
            SQLCustomTools.plot_graph,
            SQLCustomTools.list_database(),
            SQLCustomTools.info_database(),
            SQLCustomTools.fix_sql_query,
        ]
        if isinstance(SQLAgent.get_llm(), AzureChatOpenAI):
            return cls._create_agent_executor(tools, create_openai_tools_agent)
        else:
            return cls._create_agent_executor(tools, create_tool_calling_agent)

    @staticmethod
    def _create_prompt() -> ChatPromptTemplate:
        """Helper method to create a chat prompt."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", SQL_AGENT_SYSTEM_PROMPT),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )


class SQLUpdateQueryAgent(Agent):
    @classmethod
    def get_agent_executor(cls) -> AgentExecutor:
        tools = [
            SQLFilterUniqueValues.get_filter_column_details,
            SQLFilterUniqueValues.find_similar_proper_nouns,
            SQLFilterUniqueValues.assess_similar_proper_nouns,
        ]
        if isinstance(SQLAgent.get_llm(), AzureChatOpenAI):
            return cls._create_agent_executor(tools, create_openai_tools_agent)
        else:
            return cls._create_agent_executor(tools, create_tool_calling_agent)


class RAGSQLAgent(Agent):
    @classmethod
    def get_agent_executor(cls) -> AgentExecutor:
        tools = [
            RAGSQLTools.sql_chain,
            RAGSQLTools.rag_chain,
        ]
        if isinstance(SQLAgent.get_llm(), AzureChatOpenAI):
            return cls._create_agent_executor(tools, create_openai_tools_agent)
        else:
            return cls._create_agent_executor(tools, create_tool_calling_agent)

    @classmethod
    def _create_agent_executor(cls, tools, agent) -> AgentExecutor:
        """Overriding helper method to create the agent executor with given tools."""
        prompt = cls._create_prompt(tools)

        __agent = agent(cls.get_llm(), tools, prompt)

        agent_executor = AgentExecutor(
            agent=__agent,
            tools=tools,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True,
        )
        return agent_executor

    @staticmethod
    def _create_prompt(tools) -> ChatPromptTemplate:
        """Overriding helper method to create a chat prompt."""
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    AGENT_SUPERVISOR_PROMPT.format(
                        members=", ".join([tool.name for tool in tools])
                    ),
                ),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
