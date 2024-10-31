import re
from src.utils.custom_tools import RAGSQLTools


class LambdaFunctions:
    @staticmethod
    def strip_path(info_dict: dict) -> dict:
        """Removes any paths to saved graphs in the output string because they will be returned separately

        Args:
            info_dict (dict): all response info

        Returns:
            dict: all response info with reformatted output
        """

        # extracting dictionary headers
        intermediate_steps = info_dict["intermediate_steps"]
        output = info_dict["output"]

        # removing graph paths from the output response
        for tup in intermediate_steps:
            if getattr(tup[0], "tool") == "plot_graph":
                graph_path = tup[1]
                pattern = r"!\[.*?\]\(" + re.escape(graph_path) + r"\)"
                output = re.sub(pattern, "", output)

        info_dict["output"] = output

        return info_dict
