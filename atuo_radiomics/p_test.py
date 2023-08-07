import numpy as np
import random
import os
import pandas as pd
from scipy import stats
from collections import Counter


def count_list(input):
    if not isinstance(input, list):
        input = list(input)
    dict = {}
    for i in set(input):
        dict[i] = input.count(i)
    return dict

def p_test_categories(train_data_arr, test_data_arr):  # 用于对非连续值的卡方检验
    count1 = count_list(train_data_arr)
    count2 = count_list(test_data_arr)  # dict, 每个类别为key，统计了每个类别的次数
    categories = set(list(count1.keys()) + list(count2.keys()))
    contingency_dict = {}
    for category in categories:
        contingency_dict[category] = [count1[category] if category in count1.keys() else 0,
                                      count2[category] if category in count2.keys() else 0]

    contingency_pd = pd.DataFrame(contingency_dict)
    contingency_array = np.array(contingency_pd)
    _, p_value, _, _ = stats.chi2_contingency(contingency_array)
    return p_value

def p_lianxu(train_data_arr, test_data_arr):
    _, normal_p = stats.normaltest(np.concatenate((train_data_arr, test_data_arr), axis=0))
    if normal_p > 0.01:  # 正态分布用T检验
        _, p_value = stats.ttest_ind(train_data_arr, test_data_arr)
    else:  # P很小，拒绝假设，假设是来自正态分布，非正态分布用u检验
        _, p_value = stats.mannwhitneyu(train_data_arr, test_data_arr)
    return p_value

# group1 = pd.read_csv("/homes/syli/dataset/EC_all/model/train_clinical_feature.csv")["BMI"].values
# group2 = pd.read_csv("/homes/syli/dataset/EC_all/model/test_clinical_feature.csv")["BMI"].values
# group3 = pd.read_csv("/homes/syli/dataset/EC_all/outside/yfy_index_roi.csv")["age"].values
group1 = np.array([1]*301+[0]*96)
group2 = np.array([1]*142+[0]*28)
group3 = np.array([1]*18+[0]*10)
# f_value, p_value = stats.f_oneway(group1, group2, group3)
print(p_test_categories(group1, group2, group3))