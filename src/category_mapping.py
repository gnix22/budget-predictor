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

    # if user has mappings, add them into a separate json that config pulls from
    # update mapping path for instance
    def add_user_mappings(self):
        df = self.df
        # check for df containing columns subcategory, category, category type.
        # if existent, then check set to true
        def check_if_user_mappings(self):
            user_mappings_exist = False
            sub_cat_type_check = df.loc[:, df.columns.str.contains("cat", case=False)]
            if not sub_cat_type_check.empty():
                user_mappings_exist = True
            return user_mappings_exist
        # if true, user-defined-mappings are created
        user_mapping_bool = check_if_user_mappings(self)
        if user_mapping_bool == True:
            unique_mapping = df.drop_duplicates(subset=['Subcategory'])
            json_output = unique_mapping.set_index('Subcategory')[['Category', 'Category Type']].to_json('./config/user-defined-mapping.json',orient='index', indent=2)

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

