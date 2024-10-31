import pandas as pd
from sqlalchemy import create_engine
import yaml
import os
import shutil
import sys
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.llm import SQL_LLM, PDF_LLM


def vectorize_text_cols(df: pd.DataFrame, df_name: str, vectordb_folder: str) -> None:
    """Creates a vectordb for each object column in a pandas df. It only vectorizes unique non-null values in a column.

    Args:
        df (pd.DataFrame): The dataframe to be vectorized
        df_name (str): a name for the dataframe
        vectordb_folder (str): path to vectordb folder
    """

    # Step 1: Identify all columns of 'object' (string) datatype
    string_columns = df.select_dtypes(include=["object"]).columns

    # Step 2: Extract unique values for each string column
    for colname in string_columns:
        col_values = (
            df[colname].dropna().unique().tolist()
        )  # Extract unique non-null values as a list

        # Step 3: Convert column values into a format suitable for FAISS (already a list of strings)
        col_ls = [str(value) for value in col_values]  # Ensure all values are string

        # Step 4: Create and save the FAISS vector DB
        vectordb_path = f"{vectordb_folder}{df_name}_{colname}"
        if os.path.exists(vectordb_path):
            # delete the folder and its contents before saving the new vectordb
            shutil.rmtree(vectordb_path)

        vector_db = FAISS.from_texts(col_ls, embedding=SQL_LLM.get_embeddings_model())
        vector_db.save_local(vectordb_path)

        print(f"Vector DB saved for column '{colname}' in table '{df_name}'")


def initialize_db(sql_uri: str, vectordb_folder: str) -> None:
    """Initializes sqlite DB

    Args:
        sql_uri (str): path to db
        vectordb_folder (str): path to vectordb folder
    """

    # reading df
    df = pd.read_csv("./data/df.csv")

    # mapping table name to corresponding dfs
    sql_df_mapping = {"housing": df}

    # # remove db if exists
    # db_path = "./data/sample.db"
    # if os.path.exists(db_path):
    #     os.remove(db_path)

    try:
        engine = create_engine(sql_uri, echo=False)
        for table_name, df in sql_df_mapping.items():
            # remove leading and trailing whitespaces from column names
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(r"[/\n]", "_", regex=True)
            )

            # remove leading and trailing whitespaces from column values
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # vectorize unique values in all object columns in a df
            vectorize_text_cols(df, table_name, vectordb_folder)

            # writing df to db
            if df.shape[0] > int(1e6):
                chunk_size = (
                    100000  # Adjust this depending on your system's memory capacity
                )
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i : i + chunk_size]
                    chunk.to_sql(table_name, engine, if_exists="append", index=False)
            else:
                df.to_sql(table_name, engine, if_exists="replace", index=False)

    except Exception as e:
        print(e)

    finally:
        engine.dispose()


def initialize_unstructured_vectorstore(unstructured_vectorstore_path: str) -> None:
    """Creating an vectorstore for unstructured chunks of text

    Args:
        unstructured_vectorstore_path (str): _description_
    """
    loader = PyPDFLoader("./data/The_Last_Question.pdf")
    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, embedding=PDF_LLM.get_embeddings_model())

    # saving the vectorstore locally
    # removing the path if it already exists
    if os.path.exists(unstructured_vectorstore_path):
        shutil.rmtree(unstructured_vectorstore_path)
    vectorstore.save_local(unstructured_vectorstore_path)


if __name__ == "__main__":
    with open("./conf/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # unpacking config contents
    sql_uri = config["sql_uri"]
    vectordb_folder = config["vectordb_folder"]
    unstructured_vectorstore_path = config["pdf_vectorstore_path"]

    # initializing vectorstores and databases
    initialize_db(sql_uri, vectordb_folder)
    # initialize_unstructured_vectorstore(unstructured_vectorstore_path)
