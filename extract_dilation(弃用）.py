import os
from RJSNRadiomicsFeatureExtractor import main_run
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def crop():
    modals = ["DWI", "T1CE", "T2"]
    for case in tqdm(Path("/homes/syli/dataset/EC_all/EC_114_spacing").iterdir(), total=619):
        roi_path = case / "direct_dilation"
        roi_path = str(roi_path)
        for modal in modals:
            ori_roi = sitk.ReadImage(str(case) + f"/{modal}_roi_resampled.nii.gz")
            ori_roi_arr = sitk.GetArrayFromImage(ori_roi)
            for i in range(1, 11):
                roi = sitk.ReadImage(roi_path + f"/{modal}_roi_dilation_{i}.nii.gz")
                roi_array = sitk.GetArrayFromImage(roi)
                new_roi_array = roi_array - ori_roi_arr
                # new_roi_array = new_roi_array.astype(int)
                new_roi = sitk.GetImageFromArray(new_roi_array)
                new_roi.CopyInformation(roi)
                sitk.WriteImage(new_roi, os.path.join(str(case), f"{modal}_roi_{i}.nii.gz"))


def extracted_features(modal, store_path):
    save_path = os.path.join(store_path, modal)
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(1, 11)):
        save = os.path.join(save_path, f"{modal}_{i}_features.csv")
        main_run("/homes/syli/dataset/EC_all/outside/shenzhen ROI seg/cases_spacing", f"{modal}_trans", f"{modal}_dilation_{i}.nii.gz", save)


def split_and_merge(feature_root, modals, store_path):
    train_df = pd.read_csv("/homes/syli/dataset/EC_all/model/T1CE/train_numeric_feature.csv")[["CaseName", "label"]]
    test_df = pd.read_csv("/homes/syli/dataset/EC_all/model/T1CE/test_numeric_feature.csv")[["CaseName", "label"]]
    for modal in modals:
        for i in range(1, 11):
            feature_df = pd.read_csv(os.path.join(feature_root, modal, f"{modal}_{i}_features.csv"))
            new_train_df = pd.merge(train_df, feature_df, on=["CaseName"], validate="one_to_one")
            new_test_df = pd.merge(test_df, feature_df, on=["CaseName"], validate="one_to_one")
            assert len(train_df) + len(test_df) == len(new_train_df) + len(new_test_df), "拼接不对"
            store = os.path.join(store_path, modal, f"dilation_{i}")
            os.makedirs(store, exist_ok=True)
            new_train_df.to_csv(os.path.join(store, "train_numeric_feature.csv"), index=False)
            new_test_df.to_csv(os.path.join(store, "test_numeric_feature.csv"), index=False)


# split_and_merge("/homes/syli/dataset/EC_all/all_frame/now_2023", ["DWI", "T1CE", "T2"], "/homes/syli/dataset/EC_all/model/dilation_2023")
path = "/homes/syli/dataset/EC_all/outside/shenzhen ROI seg/dataframe"
extracted_features("T2", path)
