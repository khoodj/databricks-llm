# sql_decision_prompt = """Given the question below, classify whether an SQL query is needed on a database table(s) with {table_str} data.

#         Reply with "yes" or "no". Do not respond with more than one word.

#         <question>
#         {formulated_question}
#         </question>"""

# response_instruction_str = """You are a {industry} expert of a company. Refer to the following tables and column descriptions in the metadata (if any) to answer the user's question. Make sure you check the unique values for the TEXT columns in the tables in the process of filtering the data.

# <TABLES>{table_str}</TABLES>

# <TABLE_METADATA>{table_metadata}</TABLE_METADATA>

# <QUESTION>{formulated_question}</QUESTION>"""


# column_descriptions_dict = {
#     "oil_and_gas": {
#         "Country": "The country where the oil or gas field is located.",
#         "Region": "The larger geographical region within the country where the field is situated.",
#         "Sub-Region": "A smaller, more specific area within the region where the field is located.",
#         "Business Regions": "Designated business areas or operational regions used by the company for administrative or business purposes.",
#         "Basin": "The sedimentary basin where the oil or gas field is found, typically a geological low point where sediments have accumulated.",
#         "Well Name / Field Name": "The specific name of the oil or gas field or the well within the field.",
#         "UWI": "Unique Well Identifier, a standardized code used to identify a specific well within a field. (e.g.: Campos, Reconcavo Basin, Sergipe-Alagoas Basin)",
#         "Latitude": "The latitude coordinate of the well or field, indicating its position on the Earth's surface.",
#         "Longitude": "The longitude coordinate of the well or field, indicating its position on the Earth's surface.",
#         "Age": "The geological age of the rock formation where the oil or gas is located, based on geological time periods.",
#         "General Rock-Type (Clastic / Carbonate/ Volcanics/Undifferentiated)": "The general classification of the rock type found at the field (e.g.: clastic, carbonate, volcanic, or undifferentiated).",
#         "Lithology": "Detailed description of the rock composition and characteristics in the field, such as sandstone, shale, limestone, etc.",
#         "Data Source": "The origin of the data, indicating where the information about the field was obtained (e.g., external, internal).",
#         "Owner": "The entity that holds the rights to the field or well. (e.g.: NEFTEX)",
#         "Assurance": "The level of confidence or verification regarding the accuracy of the data, typically indicating quality control or assurance measures.",
#     },
#     "vegetables": {
#         "Vegetable": "The name of the vegetable being referenced.",
#         "Price_per_kg": "The price of the vegetable per kilogram, typically in local currency.",
#         "Stock_kg": "The amount of the vegetable available in stock, measured in kilograms.",
#         "Origin_Country": "The country where the vegetable was grown or sourced from.",
#         "Organic": "Indicates whether the vegetable is organically grown (True/False).",
#         "Calories_per_100g": "The number of calories present in 100 grams of the vegetable.",
#     },
# }


# table_description_system_prompt = f"""Return the names of ALL the SQL table names that MIGHT be relevant to the user question. \
# The tables and their respective descriptions are:

# <TABLE_DESCRIPTION>
# oil_and_gas: Contains data about oil and gas fields. It contains the areas Petronas (an oil and gas company) operates in, the names of the basin and well/fields, location, age, rock type etc.
# vegetables: Contains data about MyVeg (a vegetables retail company). It contains the vegetables that the company sells, remaining inventory, health information about each vegetable etc.
# </TABLE_DESCRIPTION>

# Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

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

Step 1: Try to create a connection to a sqlite database engine with the create_engine function from sqlalchemy. The path to the database is "sqlite:///./data/sample.db"

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
