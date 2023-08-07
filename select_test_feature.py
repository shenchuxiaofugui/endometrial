import pandas as pd
import numpy as np
from atuo_radiomics.Metric import EstimatePrediction
import os
import joblib
from pathlib import Path
from shutil import copyfile

def save_prediction(data_df, model, predict_store_path, key, cutoff):
    # 这个用来预测和保存预测结果，返回指标的字典
    label = data_df["label"].values.astype(int)
    predict_columns = ['label', 'Pred']
    prediction = model.predict_proba(data_df.values[:, 1:])[:, 1]
    new_data = np.concatenate((label[:, np.newaxis], prediction[:, np.newaxis]), axis=1)
    predict_df = pd.DataFrame(data=new_data, index=data_df.index, columns=predict_columns)
    predict_df.to_csv(predict_store_path + f"/{key}_prediction.csv")
    metrics = EstimatePrediction(prediction, label, key, cutoff)
    return metrics, prediction


def zscore(data, zscore_df):
    data.iloc[:, 1:] = (data.iloc[:, 1:] - zscore_df.loc["mean"]) / zscore_df.loc["std"]
    return data



def external(data_path, model_path, zscore_path=''):
    test_df = pd.read_csv(data_path, index_col=0)
    if zscore_path != "":
        pass
    features = list(pd.read_csv(model_path+"/selected_train_data.csv", index_col=0))
    try:
        new_df = test_df[features]
    except:
        new_df = test_df[[i.replace("prediction", "Pred") for i in features]]
    if os.path.exists(os.path.join(model_path, "LR model.pickle")):
        classifier = joblib.load(os.path.join(model_path, "LR model.pickle"))
    else:
        classifier = joblib.load(os.path.join(model_path, "SVM model.pickle"))

    store_path = os.path.dirname(data_path)
    new_df.to_csv(store_path+"/test_data.csv")
    train_df = pd.read_csv(model_path+"/train_prediction.csv", index_col=0)
    train_metrics = EstimatePrediction(train_df["Pred"].values, train_df["label"].values.astype(int))
    print(train_metrics["Cutoff"])
    metrics, _ = save_prediction(new_df, classifier, store_path, "external", eval(train_metrics["Cutoff"]))
    metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['values'])
    metric_df.to_csv(store_path+"/yfy_external_metrics.csv")



def merge_pred():
    for label in ["LVSI", "LN"]:
        for i in ["all", "liuzhou"]:
            for modal in ["DWI", "T1CE", "T2"]:
                df = pd.read_csv(root+f"/{modal}/{label}/{i}/3T_prediction.csv")
                df.rename(columns={"Pred": f"{modal}_Pred"}, inplace=True)
                if modal == "DWI":
                    all_df = df
                else:
                    all_df = pd.merge(all_df, df, on=["CaseName", "label"])
            all_df.to_csv(root+f"/{label}_{i}.csv", index=False)


def merge_all(root, dilation):
    # 有用的，合并瘤内瘤周的
    for modal in dilation:
        liunei_df = pd.read_csv(os.path.join(root, "liunei", modal, "zscore_features.csv"))
        liuzhou_df = pd.read_csv(os.path.join(root, "liuzhou", modal, "zscore_features.csv"))
        features = [i.replace("resampled.nii", f"dilation_{dilation[modal]}") for i in list(liuzhou_df)]
        liuzhou_df.columns = features
        sum_df = pd.merge(liunei_df, liuzhou_df, on=["CaseName", "label"], validate="one_to_one")
        modal_store = os.path.join(root, "all", modal)
        os.makedirs(modal_store, exist_ok=True)
        sum_df.to_csv(os.path.join(modal_store, "zscore_features.csv"), index=False)


def combine_prediction(root, modals):
    for i in ["intra"]:
        flag = True
        for modal in modals:
            pred_df = pd.read_csv(os.path.join(root, modal, i, "yfy_external_prediction.csv"))
            pred_df.rename(columns={"Pred": f"{modal}_prediction"}, inplace=True)
            if flag:
                new_df = pred_df
                flag = False
            else:
                new_df = pd.merge(new_df, pred_df, on=["CaseName", "label"])
        os.makedirs(os.path.join(root, "combine", i), exist_ok=True)
        new_df.to_csv(os.path.join(root, "combine", i, "yfy_features.csv"), index=False)

def batch_zscore(i):
    for j in ["DWI", "T1CE", "T2"]:
        zscore_df = pd.read_csv(zscore_path+f"/{j}/dilation_{lnm_dilation[j]}/mean_std.csv", index_col=0)
        data = pd.read_csv(df_path+f"/{i}/liuzhou/{j}/{i}_test.csv", index_col=0)
        features = [i.replace("trans", f"resampled.nii") for i in list(data)]
        data.columns = features
        new_data = zscore(data, zscore_df)
        new_data.to_csv(df_path+f"/{i}/liuzhou/{j}/zscore_features.csv")


if __name__ == "__main__":
    lvsi_dilation = {"DWI": 5, "T1CE": 6, "T2": 3}
    lnm_dilation = {"DWI": 9, "T1CE": 9, "T2": 7}


    model_path = "/homes/syli/dataset/EC_all/result/LNM"
    store_path = "/homes/syli/dataset/EC_all/outside/yfy/model_2023/LNM"
    df_path = "/homes/syli/dataset/EC_all/outside/yfy"
    zscore_path = "/homes/syli/dataset/EC_all/lnm_model/liuzhou_split"
    root = ""
    print("hahah")

    # # merge_all(store_path, lnm_dilation)
    # for i in ["intra"]:
    #     for modal in ["DWI", "T1CE", "T2"]:
    #         external(os.path.join(store_path, i, modal, "test_feature.csv"),
    #                 os.path.join(model_path, modal, i))
    #         copyfile(os.path.join(store_path, i, modal, "external_prediction.csv"),
    #                 os.path.join(model_path, modal, i, "yfy_external_prediction.csv"))
    #         copyfile(os.path.join(store_path, i, modal, "yfy_external_metrics.csv"),
    #                 os.path.join(model_path, modal, i, "yfy_external_metrics.csv"))
    # combine_prediction(model_path, ["DWI", "T1CE", "T2"])
    # for i in ["intra"]:
    #     external(os.path.join(model_path, "combine", i, "yfy_features.csv"),
    #             os.path.join(model_path, "combine", i))
#external(store_path+"/combine/teat_feature.csv", model_path)










