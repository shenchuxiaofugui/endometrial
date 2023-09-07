from atuo_radiomics.Classifier import SVM, LR
import pandas as pd
from atuo_radiomics.FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief
from atuo_radiomics.auto_run import Radiomics
import os
from atuo_radiomics.Featuretype import split_feature_type, split_image_type

train_df = pd.read_csv("./atuo_radiomics/demo_train.csv", index_col=0)


# a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief],
#               [SVM, LR], savepath="/homes/syli/dataset/demo", task_num=2, max_feature_num=10)
# a.load_csv("atuo_radiomics/demo_train.csv", "atuo_radiomics/demo_test.csv")
# os.makedirs("/homes/syli/dataset/demo", exist_ok=True)
# #a.predict_save(a.train_data, a.test_data, a.savepath, True)
# a.run()
# print("successful")
# a = train_df.iloc[:,:2]
# a["test"] = [1] * 370
