CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question/statement/instruction which might reference context in the chat history, formulate a standalone question/statement/instruction which can be understood without the chat history.

Do NOT answer the question/statement/instruction itself! Just reformulate it if needed and otherwise return it as is. If it is a question, return it as a question after rephrasing. If it is a statement, return it as a statement after rephrasing and so on."""

GENERIC_SYSTEM_PROMPT = "You are a helpful assistant."

SQL_AGENT_SYSTEM_PROMPT = (
    GENERIC_SYSTEM_PROMPT
    + """ Pay attention to the following DOs and DON'Ts:
    
    1. Answer the user's question with the given tools. If the tools yield no results, just say that you don't know. Do NOT answer the user's question with your own knowledge. 
    2. Use the tools to generate their intended outputs. Do not try to perform the tasks of the tools by yourself. 
    3. Do NOT change the SQL command when passing between the tools!"""
)

ASSESS_RELEVANT_FILTERS_PROMPT = """Given the user's question, the existing SQL query and distinct values in the column "{filter_column_name}", assess what are the appropriate proper nouns to filter in the SQL query so that the user's question can be answered. You have the discretion to modify the SQL query by adding, removing or maintaining the filtered value in column "{filter_column_name}" of the SQL query. Do NOT modify other parts of the SQL query except for the WHERE clause filter on column "{filter_column_name}"."

<INPUTS>
User question: {question}
SQL Query: {sql_query}
Distinct Values in column {filter_column_name}: {similar_unique_values_ls}
</INPUTS>

Return an appropriate SQL query. Make sure that the filtered value(s) must exist in the distinct values in column "{filter_column_name}"""

UPDATE_QUERY_PROMPT = """Check whether all exact string matches in the SQL query are correct and sufficient to answer the user's question if it were to be executed. Return an SQL query with updated exact string matches or the same SQL query if no updates are required. Do NOT modify other parts of the SQL query apart from the exact string matches.

User's question: {question}
SQL query: {sql_query}"""

RAG_INSTRUCTION_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Only use the following pieces of retrieved context to answer "
    "the question. If the question cannot be answered "
    "with the context, say that you don't know. Do NOT answer the user's question with your own knowledge."
    "\n\n"
    "<CONTEXT>\n{context}\n<\CONTEXT>"
)


AGENT_SUPERVISOR_PROMPT = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " collect the responses of each worker"
    " and respond to the user's question based on their results."
    " If all the workers cannot answer"
    " the user's question, you may seek for the user's clarification"
    " or answer with general knowedge that you know."
)

PLOT_GRAPH_INSTRUCTIONS = """Create a graph with the following steps:

Step 1: Try to create a connection to a database engine with the create_engine function from sqlalchemy. The path to the database is "{db_path}"

Step 2: Execute the sql query below and load the table into a pandas dataframe.
<SQL>
{sql_query}
</SQL>

Step 3: Create an appropriate graph using matplotlib or pandas to answer the user's question:
<QUESTION>
{question}
</QUESTION>

Step 4: Save the image in the relative folder "./data/images/" in jpg format with a unique id.

Step 5: Print the file name of the image including the RELATIVE path to it. END"""
