import json
import pandas as pd

class CategoryMapping:
    def __init__(self, df: pd.DataFrame, mapping: str = "./config/category-mapping.json"):
        self.df = df
        self.mapping_path = mapping

    def _get_mappings(self):
        # open mappings in config directory
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

    #def _check_mappings(self): Not sure if this needs to be here or later in auto classifier.
    # check if user has mapping, if not, have user decide whether or not to default,
    # uses auto classifier class to get subcategory, then append category and cat type.

