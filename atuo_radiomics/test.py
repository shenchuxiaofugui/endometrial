from Classifier import SVM, LR
import pandas as pd
import numpy as np
from Featuretype import split_feature_type
from FeatureSelector import FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief
from auto_run import Radiomics

train_df = pd.read_csv("./atuo_radiomics/demo_train.csv", index_col=0)
train_df[['label', 'LVSI']] = train_df[['LVSI', 'label']]
test_df = pd.read_csv("./atuo_radiomics/demo_test.csv", index_col=0).values[:, 2:]

a = Radiomics([FeatureSelectByRFE, FeatureSelectByANOVA, FeatureSelectByKruskalWallis, FeatureSelectByRelief],
              [SVM, LR], task_num=2)
a.load_csv("demo_train.csv", "demo_test.csv")
a.cross_validation(a.train_data, a.test_data)
print("successful")

