from Classifier import SVM
import pandas as pd

train_df = pd.read_csv("./atuo_radiomics/demo_train.csv", index_col=0)
test_df = pd.read_csv("./atuo_radiomics/demo_test.csv", index_col=0).values[:, 2:]
svm_model = SVM(train_df, tasks=2)
svm_model.fit()
a = svm_model.predict(test_df)
print(a[0].shape)