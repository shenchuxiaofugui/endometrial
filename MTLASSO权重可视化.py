import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def MTLASSOWeight():
    # path = r'X:\radiomics\Multi-task_LASSO_403\img+age+gender_correct\MTLASSO\weight.csv'
    path = r'E:\ProjectING\结果\ISMRM2023\FUFA_final_model\LR_coef.csv'
    df = pd.read_csv(path)
    feature_name = df['Feature_name']
    ep_coef = df['Coef']
    idh_coef = df['IDH']

    # x = np.arange(len(feature_name))
    x = np.arange(1, 52, 4)
    # x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    width = 0.8
    fig, ax = plt.subplots(figsize = (35, 10))
    rects1 = ax.bar(x - width/2, idh_coef, width, label='IDH mutation status')
    rects2 = ax.bar(x + width/2, ep_coef, width, label='Early recurrence')


    ax.tick_params(width=2, labelsize=20)
    ax.set_ylabel('Weight')
    ax.set_title('Feature weight')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_name)
    ax.legend(fontsize=20)
    ax.grid(axis='y')
    # ax.grid()
    fig.tight_layout()
    plt.savefig(r'E:\ProjectING\结果\MTLASSO文章\自动分割\图\weight.png', dpi=300)
    plt.show()



def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def SingleTaskWeight():
    path = r'\\mega\syli\dataset\EC_all\model\merge_log\T1CE\best_model\SVM_coef.csv'
    df = pd.read_csv(path)
    feature_name = df.iloc[0, :].tolist()
    ep_coef = df['Coef']

    x = np.arange(1, 25, 3)   # fufa
    # x = np.arange(1, 22, 3)
    # x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    width = 0.8
    fig, ax = plt.subplots(figsize=(10, 10))

    rects2 = ax.bar(x, ep_coef, width, color='crimson', label='Early recurrence')
    # rects2 = ax.bar(x, ep_coef, width, color='slateblue', label='IDH mutation status')

    # ax.tick_params(width=2, labelsize=20)
    ax.set_ylabel('Weight')
    ax.set_title('Feature weight')
    # ax.set_xticks(x)
    ax.set_xticklabels(feature_name)
    ax.legend(fontsize=20)
    ax.grid(axis='y')
    # ax.grid()
    fig.tight_layout()
    plt.savefig(r'\\mega\syli\dataset\EC_all\model\merge_log\T1CE\best_model\weight.png', dpi=300)
    plt.show()

SingleTaskWeight()