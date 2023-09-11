from atuo_radiomics.Classifier import SVM, LR
import pandas as pd
from atuo_radiomics.FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief
from atuo_radiomics.auto_run import Radiomics
import os
join = os.path.join
from atuo_radiomics.Featuretype import split_feature_type, split_image_type


root = "/homes/syli/dataset/LVSI_LNM/LNM/liunei/DWI"
a = Radiomics([FeatureSelectByANOVA],
              [SVM, LR], savepath="/homes/syli/dataset/demo", task_num=1, max_feature_num=2)
a.load_csv(join(root, "train_numeric_feature.csv"), join(root, "test_numeric_feature.csv"))
os.makedirs("/homes/syli/dataset/demo", exist_ok=True)
#a.predict_save(a.train_data, a.test_data, a.savepath, True)
a.run()
print("successful")
# a = train_df.iloc[:,:2]
# a["test"] = [1] * 370
