import pandas as pd
import numpy as np
import json
from src.category_mapping import CategoryMapping

df = pd.read_csv("./data/2019.csv")

map = CategoryMapping(df)

mapped_df = map.apply_mappings()
mapped_df["Account"] = "Checking"
mapped_df = mapped_df.loc[:, ["Account","Post Date", "Description", "Debit", "Credit", "Net", "Subcategory", "Category", "Category Type"]]

mapped_df.to_csv('reordered-2019.csv',index=False)
