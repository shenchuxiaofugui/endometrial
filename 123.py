import os.path
import numpy as np
import pandas as pd
from pathlib import Path
from Metric import EstimatePrediction
from sklearn.metrics import mean_squared_error


def calculate_metric(root, category):
    df = pd.read_csv(os.path.join(root, f"{category}_prediction.csv"))
    label = df["label"].values.astype(int)
    prediction = df["Pred"]
    metrics = EstimatePrediction(prediction, label, category)
    return metrics


def calculate_by_new_cutoff():
    df = pd.DataFrame(columns=["ACC"])
    for i in Path(r"\\mega\syli\dataset\EC_all\lnm_model\liuzhou_split").iterdir():
        if i.is_dir():
            for j in i.iterdir():
                pred = pd.read_csv(str(j)+"/original+log-sigma/best_model/test_prediction.csv")
                cutoff = pd.read_csv(str(j)+"/original+log-sigma/best_model/metric_info.csv", index_col=0)
                cutoff = eval(cutoff.loc["train_Cutoff", "values"])
                metrics = EstimatePrediction(pred["Pred"].values, pred["label"].values.astype(int), "test", cutoff)
                metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['values'])
                metric_df.to_csv(str(j)+"/original+log-sigma/best_model/test_metric_info.csv")
                df.loc[i.name+" "+j.name, "ACC"] = metrics["test_Acc"]
    df.to_csv(r"\\mega\syli\dataset\EC_all\lnm_model\liuzhou_split\new_info.csv")


def calculate_OR_CI(root):
    for i in ["DWI", "T1CE","T2"]:
        model_path = os.path.join(root, i, "merge")
        for j in Path(model_path).glob("*coef.csv"):
            coef = pd.read_csv(str(j))
        pred = pd.read_csv(os.path.join(model_path, "train_prediction.csv"))
        bias = len(pred)/(len(pred) - len(coef) - 1)
        RMSE = np.sqrt(mean_squared_error(pred["label"].values, pred["Pred"].values)*bias)
        coef["OR"] = [np.exp(i) for i in coef["Coef"].values]
        coef["CI min"] = [np.exp(i - 1.96*RMSE) for i in coef["Coef"].values]
        coef["CI max"] = [np.exp(i + 1.96 * RMSE) for i in coef["Coef"].values]
        coef["95% CI of OR"] = [f"{round(i, 3)}--{round(j, 3)}" for i,j in zip(coef["CI min"].values, coef["CI max"].values)]
        coef.to_csv(os.path.join(model_path, "OR CI.csv"))


calculate_OR_CI(r"C:\Users\handsome\Desktop\瘤周\result\LNM")










