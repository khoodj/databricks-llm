from langchain_community.utilities import SQLDatabase
import yaml

with open("./conf/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class LangchainDB:
    __sql_uri = config.get("sql_uri")
    db = SQLDatabase.from_uri(
        __sql_uri,
        # custom_table_info={  # adding this makes the responses a bit weird.
        #     "oil_and_gas": """The oil_and_gas table described below contains details about oil and gas fields where Petronas operates in Brazil""",
        #     "vegetables": """The vegetables table described below contains details about vegetables that a grocery store sells""",
        # },
    )
