import os
from pathlib import Path
import pandas as pd


def getinfo(single_path):
    if not os.path.exists(os.path.join(single_path, "best_model")):
        return None
    single_df = pd.read_csv(os.path.join(single_path, "best_model", "metric_info.csv"), index_col=0)
    feature_df = pd.read_csv(os.path.join(single_path, "best_model", "selected_train_data.csv"), index_col=0)
    # single_df = pd.read_csv(os.path.join(single_path, "metric_info.csv"), index_col=0)
    # feature_df = pd.read_csv(os.path.join(single_path, "selected_train_data.csv"), index_col=0)
    metrics = {"Train_AUC": single_df.loc["train_AUC", "values"],
               "Cv_val_AUC": single_df.loc["cv_val_AUC", "values"],
               "Test_AUC": single_df.loc["test_AUC", "values"],
               "Train_Acc": single_df.loc["train_Acc", "values"],
               "Test_Acc": single_df.loc["test_Acc", "values"],
               "Train_Sen":single_df.loc["train_Sen", "values"],
               "Train_Spe":single_df.loc["train_Spe", "values"],
                "Test_Sen":single_df.loc["test_Sen", "values"],
                "Test_Spe":single_df.loc["test_Spe", "values"],
               "Train_CI": single_df.loc["train_95% CIs", "values"],
               "Test_CI": single_df.loc["test_95% CIs", "values"],
               "Feature number": len(list(feature_df))-1}
    return metrics


def getallinfo(root):
    info_df = pd.DataFrame(columns=["Train_AUC", "Train_Acc", "Train_Sen", "Train_Spe", "Train_CI", "Cv_val_AUC",
                                    "Test_AUC", "Test_Acc", "Test_Sen", "Test_Spe", "Test_CI", "Feature number"])
    dirs = ""
    for image_type in ["original", "log-sigma", "wavelet"]:
        dirs += f"+{image_type}"
        if dirs == "+original":
            dirs = dirs[1:]
        store_path = os.path.join(root, dirs)
        if dirs == "original":
            for feature_type in ["firstorder", "shape", "texture"]:
                info_df.loc[feature_type] = getinfo(os.path.join(store_path, feature_type))
        else:
            info_df.loc[image_type.replace("log-sigma", "LoG")+"_firstorder"] = getinfo(os.path.join(root, image_type, "firstorder"))
            info_df.loc[image_type.replace("log-sigma", "LoG")+"_texture"] = getinfo(os.path.join(root, image_type, "texture"))
            info_df.loc[image_type.replace("log-sigma", "LoG")] = getinfo(os.path.join(root, image_type))
        info_df.loc[dirs.replace("log-sigma", "LoG")] = getinfo(store_path)
    info_df.to_csv(os.path.join(root, "info.csv"))


def getdilationinfo(root):
    info_df = pd.DataFrame(columns=["Train_AUC", "Cv_val_AUC", "Test_AUC", "Train_Acc", "Test_Acc", "Feature number"])
    for modal in Path(root).iterdir():
        if modal.is_dir():
            for dilation in modal.iterdir():
                info_df.loc[f"{modal.name}_{dilation.name}"] = getinfo(str(dilation))
    info_df.to_csv(os.path.join(root, "info.csv"))


def getsingleinfo(root):
    # 汇总一个模态的信息
    info_df = pd.DataFrame(columns=["Train_AUC", "Cv_val_AUC", "Test_AUC", "Feature number"])
    dirs = [i for i in Path(root).glob("*[0-9]")]
    dirs.sort()
    for i in dirs:
        for j in i.iterdir():
            info_df.loc[i.name+"_"+j.name] = getinfo(str(j))
    info_df.to_csv(os.path.join(root, "info.csv"))


def gettestinfo(path, key):
    info_df = pd.DataFrame(columns=["task", "modal", "liuzhou", "AUC"])
    for i in Path(path).iterdir():
        if i.is_dir():
            for j in i.iterdir():
                if j.is_dir():
                    for k in j.iterdir():
                        df = pd.read_csv(str(k)+"/test_metric.csv", index_col=0)
                        info_df.loc[len(info_df.index)] = [j.name, i.name, k.name, df.loc[f"{key}_AUC", "values"]]
    info_df.to_csv(path+"info.csv", index=False)


df = pd.DataFrame(columns=["Train_AUC", "Train_Acc", "Train_Sen", "Train_Spe", "Train_CI", "Cv_val_AUC",
                                    "Test_AUC", "Test_Acc", "Test_Sen", "Test_Spe", "Test_CI", "Feature number"])
for modal in ["ADC", "T1", "T2", "b1000", "T1CE"]:
    #getallinfo(f"/homes/syli/dataset/zj_data/model/{modal}")
    df1 = pd.read_csv(f"/homes/syli/dataset/zj_data/model/{modal}/info.csv", index_col=0)
    df.loc[modal] = df1.loc["original+LoG+wavelet"]
df.to_csv("/homes/syli/dataset/zj_data/model/info.csv")

# for modal in ["ADC", "T1", "T2", "b1000", "T1CE"]:
#     df1 = pd.read_csv(f"/homes/syli/dataset/zj_data/resegment_model/{modal}/NEW_info.csv", index_col=0)
#     df2 = pd.read_csv(f"/homes/syli/dataset/zj_data/model/{modal}/NEW_info.csv", index_col=0)
#     print(df1.loc["original+LoG+wavelet", :])
#     print(df2.loc["original+LoG+wavelet", :])
#     print("*"*20)
#getsingleinfo("/homes/syli/dataset/zj_data/model/T1CE")
#getdilationinfo(r"/homes/syli/dataset/EC_all/model/dilation_split")
#gettestinfo("/homes/syli/dataset/EC_all/outside/yfy/model", "yfy")



