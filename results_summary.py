import os
from pathlib import Path
import pandas as pd
join = os.path.join

def getinfo(single_path):
    if not os.path.exists(join(single_path, "best_model")):
        raise FileNotFoundError(f"{single_path} does not have best_model folder")
    single_df = pd.read_csv(join(single_path, "best_model", "metric_info.csv"), index_col=0)
    feature_df = pd.read_csv(join(single_path, "best_model", "selected_train_data.csv"), index_col=0)
    # single_df = pd.read_csv(join(single_path, "metric_info.csv"), index_col=0)
    # feature_df = pd.read_csv(join(single_path, "selected_train_data.csv"), index_col=0)
    all_metrics = {}
    for task in list(single_df):
        metrics = {"Train_AUC": single_df.loc["train_AUC", task],
                "Cv_val_AUC": single_df.loc["cv_val_AUC", task],
                "Test_AUC": single_df.loc["test_AUC", task],
                "Train_Acc": single_df.loc["train_Acc", task],
                "Test_Acc": single_df.loc["test_Acc", task],
                "Train_Sen":single_df.loc["train_Sen", task],
                "Train_Spe":single_df.loc["train_Spe", task],
                "Test_Sen":single_df.loc["test_Sen", task],
                "Test_Spe":single_df.loc["test_Spe", task],
                "Train_CI": single_df.loc["train_95% CIs", task],
                "Test_CI": single_df.loc["test_95% CIs", task],
                "Feature number": len(list(feature_df))-1}
        all_metrics[task] = metrics
    return all_metrics


def getallinfo(root):
    info_dfs = []
    dirs = ""
    task_list = []
    for image_type in ["original", "log-sigma"]:
        dirs += f"+{image_type}"
        if dirs == "+original":
            dirs = dirs[1:]
        store_path = join(root, dirs)
        if dirs == "original":
            for feature_type in ["firstorder", "shape", "texture"]:
                metric = getinfo(join(store_path, feature_type))
                if task_list == []:
                    task_list = list(metric.keys())
                    for i in range(len(task_list)):
                        info_dfs.append(pd.DataFrame(columns=["Train_AUC", "Train_Acc", "Train_Sen", "Train_Spe", "Train_CI", "Cv_val_AUC",
                                    "Test_AUC", "Test_Acc", "Test_Sen", "Test_Spe", "Test_CI", "Feature number"]))
                for i, task in enumerate(task_list):
                    info_dfs[i].loc[feature_type] = metric[task]

        else:
            for feature_type in ["firstorder", "", "texture"]:
                metric = getinfo(join(root, image_type, feature_type))
                for i, task in enumerate(task_list):
                    info_dfs[i].loc[image_type.replace("log-sigma", "LoG")+f"_{feature_type}"] = metric[task]
        for i, task in enumerate(task_list):
            info_dfs[i].loc[dirs.replace("log-sigma", "LoG")] = metric[task]
    with pd.ExcelWriter(join(root, "info.xlsx")) as writer:
        for info_df, task in zip(info_dfs, task_list):
            info_df.to_excel(writer, sheet_name=task)


def getsuminfor(root): 
    first = True
    for task in ["multi_task"]:
        for modal in ["DWI", "T1CE", "T2"]:
            df = pd.read_excel(join(root, task, "liunei", modal, "info.xlsx"), index_col=0, sheet_name="LNM")
            if first:
                first = False
                info_df = pd.DataFrame(columns=list(df))
            info_df.loc[f"{task}_{modal}"] = df.loc["original+LoG", :]
    info_df.to_csv(join(root, f"buchonginfo.csv"))


def getdilationinfo(root):
    info_df = pd.DataFrame(columns=["Train_AUC", "Cv_val_AUC", "Test_AUC", "Train_Acc", "Test_Acc", "Feature number"])
    for modal in Path(root).iterdir():
        if modal.is_dir():
            for dilation in modal.iterdir():
                info_df.loc[f"{modal.name}_{dilation.name}"] = getinfo(str(dilation))
    info_df.to_csv(join(root, "info.csv"))


def gettestinfo(path, key, task):
    info_df = pd.DataFrame(columns=["task", "modal", "liuzhou", "AUC"])
    for i in Path(path).iterdir():
        if i.is_dir():
            for j in i.iterdir():
                if j.is_dir():
                    for k in j.iterdir():
                        df = pd.read_csv(str(k)+"/test_metric.csv", index_col=0)
                        info_df.loc[len(info_df.index)] = [j.name, i.name, k.name, df.loc[f"{key}_AUC", task]]
    info_df.to_csv(path+"info.csv", index=False)



# for modal in ["DWI", "T1CE", "T2"]:
#     getallinfo(f"/homes/syli/dataset/LVSI_LNM/multi_task/liunei/{modal}")
getsuminfor("/homes/syli/dataset/LVSI_LNM")
#getsingleinfo("/homes/syli/dataset/zj_data/model/T1CE")
#getdilationinfo(r"/homes/syli/dataset/EC_all/model/dilation_split")
#gettestinfo("/homes/syli/dataset/EC_all/outside/yfy/model", "yfy")



