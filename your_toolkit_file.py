########################
# Import Dependencies
########################
import os
import re
import math
import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Type
from difflib import SequenceMatcher

# LangChain imports
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_core.tools import BaseToolkit, BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun

# Database related imports
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
    BaseSQLDatabaseTool
)
from langchain_community.agent_toolkits import create_sql_agent

# Azure OpenAI imports
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

########################
# Utility Functions
########################
def jaccard_similarity(a: str, b: str) -> float:
    """
    Calculate the Jaccard similarity between two strings.

    Args:
        a (str): First string
        b (str): Second string

    Returns:
        float: Jaccard similarity score between 0 and 1
    """
    set_a = set(a.lower())
    set_b = set(b.lower())
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union != 0 else 0

def extract_english_words(input_string: str) -> List[str]:
    """
    Extract English words from a string representation of a tuple list.

    Args:
        input_string (str): String representation of tuple list

    Returns:
        List[str]: List of extracted English words
    """
    # Convert string representation to a list
    tuple_list = eval(input_string)

    # Extract terms from each tuple
    terms = [str(item[0]) for item in tuple_list]

    # Regular expression to match only English words
    word_pattern = re.compile(r'^[a-zA-Z]+$')

    # Filter to keep only English words
    return [term for term in terms if word_pattern.match(term)]

########################
# Input Schema Classes
########################
class _ColumnValidationToolInput(BaseModel):
    """Schema for column validation tool input."""
    query: str = Field(
        ...,
        description="A string containing the table name, column name, and value to compare, separated by commas. Example: 'table_name, column_name, value_to_compare'"
    )

class _DistanceCalculationToolInput(BaseModel):
    """Schema for distance calculation tool input."""
    query: str = Field(
        ...,
        description="A string containing table name, latitude column, longitude column, reference latitude, reference longitude, and radius, all separated by commas. Example: 'properties,Latitude,Longitude,40.7128,-74.0060,5'"
    )

class _TableContextToolInput(BaseModel):
    """Schema for table context tool input."""
    query: str = Field(
        ...,
        description="Name of the table to analyze. Example: 'housing'"
    )

########################
# Custom Tool Classes
########################
class ColumnValidationTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for validating column values using Jaccard similarity."""

    name: str = "sql_db_column_validation"
    description: str = "Validate a value against a column in a table using Jaccard similarity. Input should be 'table_name, column_name, value_to_compare'."
    args_schema: Type[BaseModel] = _ColumnValidationToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Validate the given value against the column values using Jaccard similarity.

        Args:
            query (str): Input string containing table name, column name, and value
            run_manager (Optional[CallbackManagerForToolRun]): Callback manager

        Returns:
            str: Validation result message
        """
        try:
            # Parse input
            parts = [part.strip() for part in query.split(',')]
            if len(parts) != 3:
                raise ValueError("Input should be 'table_name, column_name, value_to_compare'")

            table_name, column_name, value_to_compare = parts

            # Validate table exists
            if table_name not in self.db.get_usable_table_names():
                return f"Error: The table '{table_name}' does not exist in the database."

            # Get distinct values from column
            sql_query = f"SELECT DISTINCT {column_name} FROM {table_name} LIMIT 1000"
            result = self.db.run_no_throw(sql_query)

            if isinstance(result, str):
                distinct_values = extract_english_words(result)
                highest_similarity = 0
                most_similar_term = None

                # Find most similar term
                for value in distinct_values:
                    similarity = jaccard_similarity(value_to_compare, value)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_term = value

                if highest_similarity > 0.5:
                    return f"Highest similarity match found. '{value_to_compare}' is most similar to existing value '{most_similar_term}' with Jaccard similarity of {highest_similarity:.2f}"
                else:
                    return f"No similar values found above the threshold of 0.5 for '{value_to_compare}' in '{column_name}' of '{table_name}'."

            return f"Unexpected result type: {type(result)}. Content: {str(result)}"

        except ValidationError as e:
            return f"Input validation error: {e}"
        except Exception as e:
            return f"An error occurred while validating the column value: {str(e)}"

########################
# Distance Calculation Tool
########################
class DistanceCalculationTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for finding locations within a specified radius using Haversine formula."""

    name: str = "sql_db_distance_calculation"
    description: str = (
        "Find locations within a specified radius from reference coordinates. "
        "Input should be 'table_name,latitude_column,longitude_column,ref_latitude,ref_longitude,radius_km'"
        "Example: 'properties,Latitude,Longitude,40.7128,-74.0060,5'"
    )
    args_schema: Type[BaseModel] = _DistanceCalculationToolInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Find locations within the specified radius using Haversine formula.

        Args:
            query (str): Input string with table, columns, coordinates, and radius
            run_manager: Optional callback manager

        Returns:
            str: Results of locations within specified radius
        """
        try:
            # Parse input parameters
            parts = [part.strip() for part in query.split(',')]
            if len(parts) != 6:
                raise ValueError(
                    "Input should be 'table_name,latitude_column,longitude_column,ref_latitude,ref_longitude,radius_km'"
                )

            table_name, lat_col, lon_col, ref_lat, ref_lon, radius = parts

            # Validate table exists
            if table_name not in self.db.get_usable_table_names():
                return f"Error: The table '{table_name}' does not exist in the database."

            # Convert string inputs to float
            try:
                ref_lat = float(ref_lat)
                ref_lon = float(ref_lon)
                radius = float(radius)
            except ValueError:
                return "Error: Coordinates and radius must be valid numbers."

            # Construct SQL query with Haversine formula
            sql_query = f"""
            WITH DistanceCalculation AS (
                SELECT
                    *,
                    (6371.0 * 2 * asin(
                        sqrt(
                            pow(sin(({ref_lat} - {lat_col}) * 0.0174533 / 2), 2) +
                            cos({lat_col} * 0.0174533) *
                            cos({ref_lat} * 0.0174533) *
                            pow(sin(({ref_lon} - {lon_col}) * 0.0174533 / 2), 2)
                        )
                    )) as distance_km
                FROM {table_name}
                WHERE {lat_col} IS NOT NULL
                AND {lon_col} IS NOT NULL
            )
            SELECT
                *
            FROM DistanceCalculation
            WHERE distance_km <= {radius}
            ORDER BY distance_km;
            """

            # Execute query and handle results
            result = self.db.run_no_throw(sql_query)

            if not result or result == "[]":
                return (
                    f"No locations found within {radius}km of the reference point "
                    f"({ref_lat}, {ref_lon}) in table '{table_name}'"
                )

            return (
                f"Found locations within {radius}km radius of ({ref_lat}, {ref_lon}):\n"
                f"{result}"
            )

        except ValueError as e:
            return f"Input validation error: {e}"
        except Exception as e:
            return f"An error occurred while calculating distances: {str(e)}"

########################
# Table Context Tool
########################
class InfoSQLDatabaseTool_2(BaseSQLDatabaseTool, BaseTool):
    """
    Enhanced tool for understanding table context by analyzing text columns 
    and their distinct values.
    """

    name: str = "sql_db_schema_2"
    description: str = (
        "Analyzes a table and returns text-based columns along with their distinct values. "
        "Input should be just the table name. Output format is 'column_name: value1, value2, value3'"
        "Always identify relevant column and column value for user query"
    )
    args_schema: Type[BaseModel] = _TableContextToolInput

    def _get_column_info(
        self,
        table_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List[Dict]:
        """
        Get information about all columns in the table.

        Args:
            table_name (str): Name of the table to analyze
            run_manager: Optional callback manager

        Returns:
            List[Dict]: List of dictionaries containing column name and type
        """
        sql_query = f"""
        SELECT
            name, type
        FROM pragma_table_info('{table_name}')
        """
        result = self.db.run_no_throw(sql_query)
        if isinstance(result, str):
            try:
                columns = eval(result)
                return [{"name": col[0], "type": col[1]} for col in columns]
            except:
                return []
        return []

    def _get_distinct_values(
        self, 
        table_name: str, 
        column_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> List:
        """
        Get distinct values for a text column.

        Args:
            table_name (str): Name of the table
            column_name (str): Name of the column
            run_manager: Optional callback manager

        Returns:
            List: List of distinct values in the column
        """
        sql_query = f"""
        SELECT DISTINCT {column_name}
        FROM {table_name}
        WHERE {column_name} IS NOT NULL
        AND {column_name} != ''
        AND LENGTH(TRIM({column_name})) > 0
        ORDER BY {column_name}
        LIMIT 15
        """

        result = self.db.run_no_throw(sql_query)
        if isinstance(result, str):
            try:
                return eval(result)
            except:
                return []
        return []

    def _is_text_column(self, column_type: str) -> bool:
        """
        Check if the column is a text-based column.

        Args:
            column_type (str): SQL column type

        Returns:
            bool: True if column is text-based, False otherwise
        """
        text_types = {'char', 'text', 'varchar', 'nvarchar', 'string', 'enum'}
        return any(text_type in column_type.lower() for text_type in text_types)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Analyze table structure and content.

        Args:
            query (str): Table name to analyze
            run_manager: Optional callback manager

        Returns:
            str: Formatted string of column names and their distinct values
        """
        try:
            table_name = query.strip()

            # Verify table exists
            if table_name not in self.db.get_usable_table_names():
                return f"Error: The table '{table_name}' does not exist in the database."

            # Get column information
            columns = self._get_column_info(table_name, run_manager)
            if not columns:
                return f"Error: Could not retrieve column information for table '{table_name}'"

            # Filter for text columns only
            text_columns = [col for col in columns if self._is_text_column(col["type"])]

            if not text_columns:
                return f"No text columns found in table '{table_name}'"

            # Build output with column names and values
            output_lines = []
            for col in text_columns:
                column_name = col["name"]
                distinct_values = self._get_distinct_values(table_name, column_name, run_manager)

                if distinct_values:
                    values = [str(val[0]) for val in distinct_values]
                    output_lines.append(f"{column_name}: {', '.join(values)}")

            return "\n".join(output_lines) if output_lines else "No distinct values found in text columns"

        except Exception as e:
            return f"An error occurred while analyzing the table: {str(e)}"

########################
# Main Toolkit Class
########################
class ExtendedSQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases, including custom tools."""

    db: SQLDatabase = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        """Return string representation of SQL dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """
        Get the complete list of tools in the toolkit, including both standard and custom tools.

        Returns:
            List[BaseTool]: List of all available database tools
        """
        # 1. List SQL Database Tool
        # Used to list all available tables in the database
        list_sql_database_tool = ListSQLDatabaseTool(
            db=self.db,
            description="Input should be empty. Output is a comma separated list of tables in the database."
        )

        # 2. Info SQL Database Tool
        # Used to get schema and sample data for specified tables
        info_sql_database_tool_description = (
            "Input to this tool is a comma-separated list of tables, output is the "
            "schema and sample rows for those tables. "
            "Be sure that the tables actually exist by calling "
            f"{list_sql_database_tool.name} first! "
            "Example Input: table1, table2, table3"
        )
        info_sql_database_tool = InfoSQLDatabaseTool(
            db=self.db,
            description=info_sql_database_tool_description
        )

        # 3. Enhanced Info SQL Database Tool
        # Used for deeper analysis of table context and column values
        understand_table_context_tool = InfoSQLDatabaseTool_2(
            db=self.db,
            description=(
                "Provides the schema and all column names and their unique values for specified tables. "
                "Always use this tool when you need more context on database, such as checking which "
                "column needed to be used to answer questions. "
                "Be sure that the tables actually exist by calling "
                f"{list_sql_database_tool.name} first!"
            )
        )

        # 4. Query SQL Database Tool
        # Used to execute SQL queries
        query_sql_database_tool_description = (
            "Input to this tool is a detailed and correct SQL query, output is a "
            "result from the database. If the query is not correct, an error message "
            "will be returned. If an error is returned, rewrite the query, check the "
            "query, and try again. If you encounter an issue with Unknown column "
            f"'xxxx' in 'field list', use {info_sql_database_tool.name} "
            "to query the correct table fields."
        )
        query_sql_database_tool = QuerySQLDataBaseTool(
            db=self.db,
            description=query_sql_database_tool_description
        )

        # 5. Query SQL Checker Tool
        # Used to validate SQL queries before execution
        query_sql_checker_tool_description = (
            "Use this tool to double check if your query is correct before executing "
            "it. Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}!"
        )
        query_sql_checker_tool = QuerySQLCheckerTool(
            db=self.db,
            llm=self.llm,
            description=query_sql_checker_tool_description
        )

        # 6. Column Validation Tool
        # Used to validate column values using similarity matching
        column_validation_tool_description = (
            "Use this tool when you have created a SQL query with WHERE clause, but "
            "after executing SQL, the result is None or no result or error, and you "
            "want to validate if the column values are correct for specific column name. "
            "Input should be a table name, column name and column values to check. "
            "Always use this tool before executing a query with "
            f"{query_sql_database_tool.name}"
        )
        column_validation_tool = ColumnValidationTool(
            db=self.db,
            description=column_validation_tool_description
        )

        # 7. Distance Calculation Tool
        # Used for geospatial queries
        distance_calculation_tool = DistanceCalculationTool(
            db=self.db,
            description=(
                "Use this tool to find properties within a specified radius from reference coordinates. "
                "Input should be 'table_name,latitude_column,longitude_column,ref_latitude,ref_longitude,radius_km'. "
                "Example: 'properties,Latitude,Longitude,40.7128,-74.0060,5' will find locations within 5km of the reference point."
            )
        )

        # Return all tools in the preferred order of use
        return [
            # Query tools
            query_sql_database_tool,    # For executing SQL queries
            query_sql_checker_tool,     # For validating queries
            
            # Information tools
            list_sql_database_tool,     # For listing available tables
            info_sql_database_tool,     # For basic schema information
            understand_table_context_tool,  # For detailed schema analysis
            
            # Specialized tools
            column_validation_tool,     # For column value validation
            distance_calculation_tool,  # For geospatial queries
        ]

    def get_context(self) -> dict:
        """
        Return database context that may be needed in agent prompt.
        
        Returns:
            dict: Database context information
        """
        return self.db.get_context()


def create_sql_agent_with_extra_tools(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    **kwargs: Any
) -> Any:
    toolkit = ExtendedSQLDatabaseToolkit(db=db, llm=llm)

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True
    )

########################
# Main Execution
########################
def main():
    # Set up Azure OpenAI credentials
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

    # Load and prepare data
    df = pd.read_csv('df.csv')
    df.set_index('Id', inplace=True)

    # Set up SQLite database
    current_directory = os.getcwd()
    db_filename = 'housing.db'
    db_path = os.path.join(current_directory, db_filename)

    # Create database connection and save DataFrame
    with sqlite3.connect(db_path) as conn:
        df.to_sql('housing', conn, if_exists='replace', index=True)

    # Initialize database connection
    db_local = SQLDatabase.from_uri(f"sqlite:///{db_path}")

    # Create and initialize agent
    agent_executor = create_sql_agent_with_extra_tools(
        llm=llm,
        db=db_local,
        handle_parsing_errors=True
    )

    # Example query
    thought_response = agent_executor.invoke("Find Facility within 5km of coordinates 2.8,101.5")
    print(thought_response)

if __name__ == "__main__":
    main()