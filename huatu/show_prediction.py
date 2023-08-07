import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(16,4))
k=1
for i in ["DWI", "T1CE", "T2"]:
    for j in ["intra", "peri"]:
        new_df = pd.read_csv(f'C:/Users/handsome/Desktop/瘤周/result/LVSI/{i}/{j}/test_prediction.csv')
        data = new_df["Pred"].values
        pic2 = fig.add_subplot(2, 4, k)
        plt.hist(data, bins=10)
        plt.title(i+" "+j)
        plt.legend([f"median: {np.round(np.median(data), 3)}"])
        k+=1
new_df = pd.read_csv(f'C:/Users/handsome/Desktop/瘤周/result/LVSI/combine/merge/test_prediction.csv')
data = new_df["Pred"].values
pic2 = fig.add_subplot(2, 4, k)
plt.hist(data, bins=10)
plt.title("Combined")
plt.legend([f"median: {np.round(np.median(data), 3)}"])
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()