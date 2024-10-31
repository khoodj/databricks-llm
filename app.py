import os

os.environ["CURL_CA_BUNDLE"] = ""

import uuid
import streamlit as st
from dotenv import load_dotenv
from src.pipelines.sql_chain import SQLChain
from typing import *
import pandas as pd

load_dotenv()

from langfuse.callback import CallbackHandler

IMG_FOLDER_URI = "./data/images"

st.title("Chat with SQL Database")


@st.cache_resource
def initialize_sql_chain():
    return SQLChain().output_chain()


@st.cache_resource
def get_uuid():
    return str(uuid.uuid4()), str(uuid.uuid4())


# initializing pdf and sql chain
sql_chain = initialize_sql_chain()
rag_session_uuid, sql_session_uuid = get_uuid()
sql_langfuse_handler = CallbackHandler(session_id=sql_session_uuid)

# Initialize session state for both chat interfaces
if "sql_chat_messages" not in st.session_state:
    st.session_state.sql_chat_messages = []


def generate_response(query) -> Tuple[List[Optional[str]], str]:
    graph_id_ls = []

    # generating response
    sql_chain_response = sql_chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": sql_session_uuid},
            "callbacks": [sql_langfuse_handler],
        },
    )

    print("sql_chain_response:\n", sql_chain_response)

    # extracting output from the response
    output = sql_chain_response.get("output", "")

    # extracting image from the response
    for tup in sql_chain_response["intermediate_steps"]:
        if getattr(tup[0], "tool") == "plot_graph":
            graph_id_ls.append(tup[1])

    return graph_id_ls, output


# Select the appropriate message list based on the selected chat interface
messages = st.session_state.sql_chat_messages

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["image_path_ls"]:
            for img_uri in message["image_path_ls"]:
                try:
                    st.image(img_uri, caption="Sample Image")
                except Exception as e:
                    st.error(f"Error in displaying graph: {e}")

if prompt := st.chat_input("What is your question?"):
    st.chat_message("user").markdown(prompt)
    messages.append({"role": "user", "content": prompt, "image_path_ls": []})

    graph_id_ls, response = generate_response(prompt)
    IMG_URI_LS = graph_id_ls

    with st.chat_message("assistant"):
        st.markdown(response)
        if IMG_URI_LS:
            for img_uri in IMG_URI_LS:
                try:
                    st.image(img_uri, caption="Sample Image")
                except Exception as e:
                    st.error(f"Error in displaying graph: {e}")
    messages.append(
        {
            "role": "assistant",
            "content": response,
            "image_path_ls": IMG_URI_LS if IMG_URI_LS else None,
        }
    )

# Update the session state with the new messages
st.session_state.sql_chat_messages = messages
# Uploader page for CSV files
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Only CSV files are accepted.")

        st.write("File uploaded successfully!")
        st.write(df.head())
    except Exception as e:
        st.error(f"Error: {e}")
