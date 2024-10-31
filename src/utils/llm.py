import yaml
from abc import ABC, abstractmethod
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

with open("./conf/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class LLM(ABC):
    @classmethod
    @abstractmethod
    def get_llm():
        """To get the llm for a specific purpose"""
        pass

    @classmethod
    @abstractmethod
    def get_embeddings_model():
        """To get the embeddings model for a specific purpose"""
        pass


class SQL_LLM(LLM):
    __sql_azure_endpoint = config.get("sql_azure_endpoint")
    __sql_openai_api_type = config.get("sql_openai_api_type")
    __sql_openai_api_version = config.get("sql_openai_api_version")
    __sql_llm_deployment_name = config.get("sql_deployment_llm_name")
    __sql_embedding_deployment_name = config.get("sql_deployment_embeddings_name")
    __temperature = config.get("sql_temperature")
    __chunk_size = config.get("sql_chunk_size")

    @classmethod
    def get_llm(cls):
        llm = AzureChatOpenAI(
            azure_endpoint=cls.__sql_azure_endpoint,
            openai_api_type=cls.__sql_openai_api_type,
            openai_api_version=cls.__sql_openai_api_version,
            deployment_name=cls.__sql_llm_deployment_name,
            temperature=cls.__temperature,
        )
        return llm

    @classmethod
    def get_embeddings_model(cls):
        embeddings_model = AzureOpenAIEmbeddings(
            openai_api_type=cls.__sql_openai_api_type,
            model=cls.__sql_embedding_deployment_name,
            deployment=cls.__sql_embedding_deployment_name,
            chunk_size=cls.__chunk_size,
        )
        return embeddings_model


class PDF_LLM(LLM):
    __pdf_azure_endpoint = config.get("pdf_azure_endpoint")
    __pdf_openai_api_type = config.get("pdf_openai_api_type")
    __pdf_openai_api_version = config.get("pdf_openai_api_version")
    __pdf_llm_deployment_name = config.get("pdf_deployment_llm_name")
    __pdf_embedding_deployment_name = config.get("pdf_deployment_embeddings_name")
    __temperature = config.get("pdf_temperature")
    __chunk_size = config.get("pdf_chunk_size")

    @classmethod
    def get_llm(cls):
        llm = AzureChatOpenAI(
            azure_endpoint=cls.__pdf_azure_endpoint,
            openai_api_type=cls.__pdf_openai_api_type,
            openai_api_version=cls.__pdf_openai_api_version,
            deployment_name=cls.__pdf_llm_deployment_name,
            temperature=cls.__temperature,
        )
        return llm

    @classmethod
    def get_embeddings_model(cls):
        embeddings_model = AzureOpenAIEmbeddings(
            openai_api_type=cls.__pdf_openai_api_type,
            model=cls.__pdf_embedding_deployment_name,
            deployment=cls.__pdf_embedding_deployment_name,
            chunk_size=cls.__chunk_size,
        )
        return embeddings_model


class PDF_SQL_LLM(PDF_LLM):
    pass
