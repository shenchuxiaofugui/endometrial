import pandas as pd
import os


def single_modal_merge(best_dilation):
    for split in ["train", "test"]:
        for modal in ["DWI", "T2", "T1CE"]:
            cancer_df = pd.read_csv(f"/homes/syli/dataset/EC_all/lnm_model/liunei/{modal}/original+log-sigma/best_model/selected_{split}_data.csv")
            dilation_df = pd.read_csv(f"/homes/syli/dataset/EC_all/lnm_model/liuzhou/{modal}/dilation_{best_dilation[modal]}/best_model/selected_{split}_data.csv")
            dilation_features = [i.replace("resampled.nii", f"dilation_{best_dilation[modal]}") for i in list(dilation_df)]
            dilation_df.columns = dilation_features
            store_path = f"/homes/syli/dataset/EC_all/lnm_model/merge_liu_zhou/{modal}"
            os.makedirs(store_path, exist_ok=True)
            df = pd.merge(cancer_df, dilation_df, on=["CaseName", "label"])
            df.to_csv(os.path.join(store_path, f"{split}_numeric_feature.csv"), index=False)


def combine_prediction(root, modals, store, best_dilation={}):
    for split in ["train", "test"]:
        flag = True
        for modal in modals:
            pred_df = pd.read_csv(os.path.join(root, f"{modal}/original+log-sigma/best_model/{split}_prediction.csv"))[["CaseName", "label", "Pred"]] #dilation_{best_dilation[modal]}/
            pred_df.rename(columns={"Pred": f"{modal}_prediction"}, inplace=True)
            if flag:
                new_df = pred_df
                flag = False
            else:
                new_df = pd.merge(new_df, pred_df, on=["CaseName", "label"])
        new_df.to_csv(os.path.join(store, f"{split}_numeric_feature.csv"), index=False)


combine_prediction("/homes/syli/dataset/EC_all/lnm_model/liunei", ["DWI", "T1CE", "T2"], "/homes/syli/dataset/EC_all/lnm_model/combine_2023/combine_liunei")
#single_modal_merge()




