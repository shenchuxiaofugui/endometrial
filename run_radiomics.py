import numpy as np
from atuo_radiomics.auto_run import Radiomics
from atuo_radiomics.Classifier import LR, SVM
from atuo_radiomics.FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief
import os
join = os.path.join
from pathlib import Path
import shutil
import pandas as pd

LVSI_dilation = {"DWI": 5, "T1CE": 6, "T2": 3}
LNM_dilation = {"DWI": 9, "T1CE": 9, "T2": 7}


def run_dilation(root, modal, tasks=1, min_i=1, max_i=11):
    for i in range(min_i, max_i):
        path = os.path.join(root, modal, f"dilation_{i}")
        a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief],
                     [LR, SVM], path, max_feature_num=10, task_num = tasks, has_shape=False)
        a.load_csv(os.path.join(path, "train_numeric_feature.csv"), os.path.join(path, "test_numeric_feature.csv"))
        a.run()


def merge_liuzhou():
    best_dilation = {"DWI": 5, "T1CE": 6, "T2": 3}
    # best_dilation = {"DWI": 9, "T1CE": 9, "T2": 7}
    for category in ["train", "test"]:
        for modal in ["DWI", "T2", "T1CE"]:
            cancer_df = pd.read_csv(
                f"/homes/syli/dataset/EC_all/model/{modal}/original+log-sigma/best_model/selected_{category}_data.csv")
            dilation_df = pd.read_csv(
                f"/homes/syli/dataset/EC_all/model/dilation_split/{modal}/dilation_{best_dilation[modal]}/original+log-sigma/best_model/selected_{category}_data.csv")
            dilation_features = [i.replace("resampled.nii", f"dilation_{best_dilation[modal]}") for i in list(dilation_df)]
            dilation_df.columns = dilation_features
            store_path = f"/homes/syli/dataset/EC_all/model/merge_all/merge_features/{modal}"
            os.makedirs(store_path, exist_ok=True)
            df = pd.merge(cancer_df, dilation_df, on=["CaseName", "label"])
            df.to_csv(os.path.join(store_path, f"{category}_numeric_feature.csv"), index=False)



def copy_new_dilation(source):
    for i in Path(source).iterdir():
        if not i.is_dir():
            continue
        for j in i.iterdir():
            tar_path = str(j).replace("liuzhou", "liuzhou_split")
            os.makedirs(tar_path, exist_ok=True)
            shutil.copy(str(j)+"/train_numeric_feature.csv", tar_path+"/train_numeric_feature.csv")
            shutil.copy(str(j) + "/test_numeric_feature.csv", tar_path + "/test_numeric_feature.csv")


def merge_label_feature(clinical_path, feature_path, modals, key, store_path):
    clinical_df = pd.read_excel(clinical_path)
    for modal in modals:
        df = pd.read_csv(os.path.join(feature_path, f"{modal}_features.csv"))
        new_df = pd.merge(clinical_df[["CaseName", key]], df)
        new_df.rename(columns={key: "label"}, inplace=True)
        os.makedirs(os.path.join(store_path, modal), exist_ok=True)
        print(os.path.join(store_path, modal))
        new_df.to_csv(os.path.join(store_path, modal, f"{key}_test.csv"), index=False)





if __name__ == "__main__":
    root = "/homes/syli/dataset/LVSI_LNM/multi_task/liuzhou"
    #
    # merge_label_feature(root+"/shenzhen.xlsx", root+"/shenzhen ROI seg/dataframe", ["DWI", "T1CE", "T2"], 
    #                     "LNM", root+"/shenzhen ROI seg/LNM/liunei")
    for modal in ["T2"]:
        run_dilation(root, modal, 2, 2, 11)
        # a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief],
        #             [SVM, LR], savepath=root, task_num=1, max_feature_num=10)
        # a.load_csv(join(root, "train_numeric_feature.csv"), join(root, "test_numeric_feature.csv"))
        # #a.predict_save(a.train_data, a.test_data, a.savepath, True)
        # a.run()
    # combine_prediction(root, ["T1", "T2", "T1CE", "ADC", "b1000"], root)
    # external_test()


