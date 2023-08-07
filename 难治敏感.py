import pandas as pd
from xpinyin import Pinyin

label_df = pd.read_excel("/homes/syli/dataset/neimo/new/MRI_list.xlsx")
data_df = pd.read_excel("/homes/syli/dataset/neimo/new/2013-2018EC.xlsx")[["ID", "姓名"]]
df = pd.merge(data_df, label_df, on=["姓名"])
print(len(label_df), len(df))