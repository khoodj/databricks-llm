import streamlit as st
import os
import sqlite3
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_community.utilities import SQLDatabase
from typing import Any

# Import the ExtendedSQLDatabaseToolkit and related functions
from your_toolkit_file import ExtendedSQLDatabaseToolkit, create_sql_agent_with_extra_tools

# Set page config
st.set_page_config(
    page_title="SQL Database Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        text-align: right;
    }
    .assistant-message {
        background-color: #f0f0f0;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)


api_key = os.environ.get('API_KEY')
azure_endpoint = os.environ.get('AZURE_ENDPOINT')

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version="2024-03-01-preview",
    azure_deployment="myTalentX_GPTo",
    temperature=0
)

# Set up SQLite database
current_directory = os.getcwd()
db_filename = 'housing.db'
db_path = os.path.join(current_directory, db_filename)

# Initialize database connection
db_local = SQLDatabase.from_uri(f"sqlite:///{db_path}")

# Create and initialize agent
agent_executor = create_sql_agent_with_extra_tools(
    llm=llm,
    db=db_local,
)


def main():
    st.title("SQL Database Assistant 🤖")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your SQL Database Assistant. I can help you query and analyze the database. How can I assist you today?"
        }]

    # Add a clear chat history button
    if st.button('Clear Chat History'):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello! I'm your SQL Database Assistant. I can help you query and analyze the database. How can I assist you today?"
        }]
        st.write("Chat history cleared.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            alignment = "flex-end" if message["role"] == "user" else "flex-start"
            st.markdown(
                f"""
                <div style='display: flex; justify-content: {alignment}; margin-bottom: 10px;'>
                    <strong>{message["content"]}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )

    # Get user input
    if user_query := st.chat_input("Ask me about the database...", key="user_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response from agent
                response = agent_executor.invoke(user_query)

                
                # Add assistant response to chat history
                assistant_message = {
                    "role": "assistant",
                    "content": response["output"]
                }
                st.write(response["output"])
                st.session_state.messages.append(assistant_message)

if __name__ == "__main__":
    main()