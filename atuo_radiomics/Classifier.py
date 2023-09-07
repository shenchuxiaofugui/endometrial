# -*- coding: utf-8 -*-
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


class Classifier:
    def __init__(self, tasks, dataframe, model):
        self.dataframe = dataframe
        self._x = dataframe.iloc[:, tasks:]
        self._y = dataframe.iloc[:, :tasks].values
        if tasks == 1:
             self._y = np.ravel(self._y)
        self.tasks = tasks
        self.y_name = list(dataframe)[:tasks]
        self.__model = model
        if self.tasks != 1:
            self.__model = MultiOutputClassifier(self.__model)
        self.fit()

    def set_model(self, model):
        self.__model = model
        if self.tasks != 1:
            self.__model = MultiOutputClassifier(self.__model)

    def get_model(self):
        if self.tasks == 1:
            return [self.__model]
        else:
            return [i for i in self.__model.estimators_]

    def predict(self, x, is_probability=True):
        if is_probability:
            if self.tasks == 1:
                return {"label_pred": self.__model.predict_proba(x)[:, 1]}
            else:
                prediction = self.__model.predict_proba(x)
                # 做成字典的形式 
                d = dict()
                for task, pred in zip(self.y_name, prediction):
                    d[task+"_pred"] = pred[:,1]
                return d

        else:
            return self.__model.Predict(x)
        
    def save(self, store_folder):
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        if not os.path.isdir(store_folder):
            print('The store function of SVM must be a folder path')
            return
        if os.path.isdir(store_folder):
            store_path = os.path.join(store_folder, f'{self.get_name()} model.pickle')
            with open(store_path, 'wb') as f:
                d = dict()
                for key, model in zip(self.y_name, self.get_model()):
                    d[key] = model
                pickle.dump(d, f)

        # Save the coefficients
        try:
            coef_path = os.path.join(store_folder, f'{self.get_name()}_coef.csv')
            flag = True
            for feature, model in zip(self.y_name, self.get_model()):
                if flag:
                    df = pd.DataFrame(data=np.transpose(model.coef_),
                                    index=self.dataframe.columns.tolist()[self.tasks:], columns=[feature+'_coef'])
                    df.loc["intercept", feature+'_coef'] = model.intercept_.reshape(1, 1)
                    flag = False
                else:
                    new_df = pd.DataFrame(data=np.transpose(model.coef_),
                                    index=self.dataframe.columns.tolist()[self.tasks:], columns=[feature+'_coef'])
                    new_df.loc["intercept", feature+'_coef'] = model.intercept_.reshape(1, 1)
                    df = df.join(new_df)
            df.to_csv(coef_path)
        except Exception as e:
            content = f'{self.get_name()} with specific kernel does not give intercept: '
            print('{} \n{}'.format(content, e.__str__()))    

    def fit(self):
        self.__model.fit(self._x, self._y)


class SVM(Classifier):
    def __init__(self, dataframe, tasks=1, **kwargs):
        if 'kernel' not in kwargs.keys():
            kwargs['kernel'] = 'linear'
        if 'C' not in kwargs.keys():
            kwargs['C'] = 1.0
        if 'probability' not in kwargs.keys():
            kwargs['probability'] = True
        super(SVM, self).__init__(tasks, dataframe, (SVC(random_state=0, **kwargs)))


    def get_name(self):
        return "SVM"


class LR(Classifier):
    def __init__(self, dataframe, tasks=1, **kwargs):
        if 'solver' in kwargs.keys():
            LR_model = LogisticRegression(**kwargs)
        else:
            LR_model = LogisticRegression(solver='saga', tol=0.01,
                                                         random_state=0, **kwargs)
        self.name = "LR"
        super(LR, self).__init__(tasks, dataframe, LR_model)


    def get_name(self):
        return "LR"


class LRLasso(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(LRLasso, self).__init__()
        if 'solver' in kwargs.keys():
            super(LRLasso, self).set_model(LogisticRegression(penalty='l1', **kwargs))
        else:
            super(LRLasso, self).set_model(LogisticRegression(penalty='l1', solver='liblinear',
                                                              random_state=0, **kwargs))
        self._x = np.array(dataframe.values[:, 1:])
        self._y = np.array(dataframe['label'].tolist())
        self.dataframe = dataframe
        self.fit()

    def save(self, store_folder):
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        if not os.path.isdir(store_folder):
            print('The store function of SVM must be a folder path')
            return
        if os.path.isdir(store_folder):
            store_path = os.path.join(store_folder, 'LR lasso model.pickle')
            with open(store_path, 'wb') as f:
                pickle.dump(self.get_model(), f)

        # Save the coefficients
        try:
            coef_path = os.path.join(store_folder, 'LR lasso_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.get_model().coef_),
                              index=self.dataframe.columns.tolist()[1:], columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'LR can not load coef: '
            print('{} \n{}'.format(content, e.__str__()))

        try:
            intercept_path = os.path.join(store_folder, 'LR lasso_intercept.csv')
            intercept_df = pd.DataFrame(data=self.get_model().intercept_.reshape(1, 1),
                                        index=['intercept'], columns=['value'])
            intercept_df.to_csv(intercept_path)
        except Exception as e:
            content = 'LR can not load intercept: '
            print('{} \n{}'.format(content, e.__str__()))


class LDA(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(LDA, self).__init__()
        super(LDA, self).set_model(LinearDiscriminantAnalysis(**kwargs))
        self._x = np.array(dataframe.values[:, 1:])
        self._y = np.array(dataframe['label'].tolist())
        self.dataframe = dataframe
        self.fit()

    def save(self, store_folder):
        if not os.path.exists(store_folder):
            os.makedirs(store_folder)
        if not os.path.isdir(store_folder):
            print('The store function of SVM must be a folder path')
            return
        if os.path.isdir(store_folder):
            store_path = os.path.join(store_folder, 'model.pickle')
            with open(store_path, 'wb') as f:
                pickle.dump(self.get_model(), f)

        # Save the coefficients
        try:
            coef_path = os.path.join(store_folder, 'LDA_coef.csv')
            df = pd.DataFrame(data=np.transpose(self.get_model().coef_),
                              index=self.dataframe.columns.tolist()[1:], columns=['Coef'])
            df.to_csv(coef_path)
        except Exception as e:
            content = 'LDA with specific kernel does not give coef: '
            print('{} \n{}'.format(content, e.__str__()))


class RandomForest(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(RandomForest, self).__init__()
        if 'n_estimators' not in kwargs.keys():
            super(RandomForest, self).set_model(RandomForestClassifier(random_state=0,
                                                                       n_estimators=200, **kwargs))
        else:
            super(RandomForest, self).set_model(RandomForestClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.values[:, 1:])
        self._y = np.array(dataframe['label'].tolist())
        self.dataframe = dataframe
        self.fit()

    def save(self, store_folder):
        pass


class DecisionTree(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(DecisionTree, self).__init__()
        super(DecisionTree, self).set_model(DecisionTreeClassifier(random_state=0, **kwargs))
        self._x = np.array(dataframe.values[:, 1:])
        self._y = np.array(dataframe['label'].tolist())
        self.dataframe = dataframe
        self.fit()

    def save(self, store_folder):
        pass


class NaiveBayes(Classifier):
    def __init__(self, dataframe, **kwargs):
        super(NaiveBayes, self).__init__()
        super(NaiveBayes, self).set_model(GaussianNB(**kwargs))
        self._x = np.array(dataframe.values[:, 1:])
        self._y = np.array(dataframe['label'].tolist())
        self.dataframe = dataframe
        self.fit()

    def save(self, store_folder):
        pass


if __name__ == '__main__':
    data_path = r'C:\Users\HJ Wang\Desktop\ML\data mining\group work\machine learning\data\train_numeric_feature.csv'
    train_df = pd.read_csv(data_path, index_col=0)
    train_df = train_df.replace(np.inf, np.nan)
    train_df = train_df.dropna(axis=1, how='any')

    test_path = r'C:\Users\HJ Wang\Desktop\ML\data mining\group work\machine learning\data\test_numeric_feature.csv'
    test_df = pd.read_csv(test_path, index_col=0)
    test_df = test_df.replace(np.inf, np.nan)
    test_df = test_df.dropna(axis=1, how='any')

    svm_modal = SVM(train_df)
    predict = svm_modal.predict(np.array(test_df.values[:, 1:]))
    print('predict:', predict)
    print('label:', np.array(test_df['label'].tolist()))

    save_path = r'.\output'
    svm_modal.save(save_path)
