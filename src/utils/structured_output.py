from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


class SQLQuery(BaseModel):
    """Extracts SQL Query from a string"""

    sql_query: str = Field(description="An SQL query")


class IsExactMatchFilterQuery(BaseModel):
    """Assess whether an SQL query contains an exact match filter in the WHERE clause. Filters with wild cards are not counted. If yes, returns True. Otherwise returns False."""

    is_exact_match_query: bool = Field(
        description="""Assess whether an SQL query contains an exact match filter filter in the WHERE clause. Filters with wild cards are not counted. If there are multiple filters, return True if any of the filters require exact matches."""
    )


class ExtractSQLQueryDetails(BaseModel):
    """Given an SQL query with a WHERE clause filter, extract the column name that is being filtered, the filtered value and the table name"""

    column_name: str = Field(
        description="""column name that is being filtered in the SQL query"""
    )

    filtered_value: str = Field(
        description="""the proper noun that is being filtered in the column"""
    )

    table_name: str = Field(description="""the table name in the SQL query""")


class FilePath(BaseModel):
    """Extracts a relative file path from a string"""

    file_path: str = Field(
        description="""A relative file path in a string. If no relative paths exist, return an empty string."""
    )
