import pandas as pd
from pathlib import Path
import os


def renew(split):
    for modal in ["DWI", "T1CE", "T2"]:
        feature_df = pd.read_csv(os.path.join(r"\\mega\syli\dataset\EC_all\model", f"{modal}", f"{split}_numeric_feature.csv"))
        del feature_df["label"]
        df = pd.read_csv(r"\\mega\syli\dataset\EC_all\clinical_data.csv")[["CaseName", "label"]]
        new_df = pd.merge(df, feature_df, on=["CaseName"], validate="one_to_one")
        assert len(new_df) == len(feature_df), "wrong"
        storepath = os.path.join(r"\\mega\syli\dataset\EC_all\lnm_model\liunei", f"{modal}")
        os.makedirs(storepath, exist_ok=True)
        new_df.to_csv(os.path.join(storepath, f"{split}_numeric_feature.csv"), index=False)


def renew_liuzhou(split):
    for modal in ["DWI", "T1CE", "T2"]:
        for i in range(1, 11):
            feature_df = pd.read_csv(
                os.path.join(r"\\mega\syli\dataset\EC_all\model\dilation_2023", f"{modal}/dilation_{i}/{split}_numeric_feature.csv"))
            del feature_df["label"]
            df = pd.read_csv(r"\\mega\syli\dataset\EC_all\clinical_data.csv")[["CaseName", "label"]]
            new_df = pd.merge(df, feature_df, on=["CaseName"], validate="one_to_one")
            assert len(new_df) == len(feature_df), "wrong"
            storepath = os.path.join(r"\\mega\syli\dataset\EC_all\lnm_model\liuzhou", f"{modal}/dilation_{i}")
            os.makedirs(storepath, exist_ok=True)
            new_df.to_csv(os.path.join(storepath, f"{split}_numeric_feature.csv"), index=False)




renew_liuzhou("test")

