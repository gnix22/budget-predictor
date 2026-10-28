import json
import pandas as pd

class CategoryMapping:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.mapping_path = "./config/category-mapping.json"

    def _get_mappings(self):
        with open(self.mapping_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data

    # add add_mappings method eventually

    def apply_mappings(self):
        mappings = self._get_mappings()
        # helper function
        def get_row_data(subcategory):
            # if not in current mapping 
            fallback = {"Category": "Unclassified", "Category Type": "Unclassified"}
            rule = mappings.get(subcategory, fallback)
            return pd.Series([rule["Category"], rule["Category Type"]])

        self.df[["Category", "Category Type"]] = self.df["Subcategory"].apply(get_row_data)
        return self.df
