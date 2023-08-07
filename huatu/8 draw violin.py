# -*- coding: utf-8 -*-
import plotly.graph_objects as go

import pandas as pd

train_df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\混淆矩阵\t2+c\train_prediction.csv')
test_df = pd.read_csv(r'C:\Users\HJ Wang\Desktop\model result T2+clinic\混淆矩阵\t2+c\test_prediction.csv')

AUC_random = data_df['external test AUC']
AUC_model = data_df['resplit external test AUC']

external_train_random = data_df['external train AUC']
external_train_model = data_df['resplit external train AUC']

internal_test_random = data_df['test AUC']
internal_test_model = data_df['resplit test AUC']

internal_train_random = data_df['train AUC']
internal_train_model = data_df['resplit train AUC']

fig = go.Figure()

fig.add_trace(go.Violin(x=['internal train']*len(internal_train_random),
                        y=internal_train_random,
                        legendgroup='random', scalegroup='random', name='random',
                        side='negative', spanmode='hard',
                        line_color='blue',
                        width=1))
fig.add_trace(go.Violin(x=['internal train']*len(internal_train_model),
                        y=internal_train_model,
                        legendgroup='resplit', scalegroup='resplit', name='resplit',
                        side='positive', spanmode='hard',
                        line_color='orange',
                        width=1))

fig.add_trace(go.Violin(x=['internal test']*len(internal_test_random),
                        y=internal_test_random,
                        legendgroup='', scalegroup='', name='',
                        side='negative', spanmode='hard',
                        line_color='blue',
                        width=1))
fig.add_trace(go.Violin(x=['internal test']*len(internal_test_model),
                        y=internal_test_model,
                        legendgroup='', scalegroup='', name='',
                        side='positive', spanmode='hard',
                        line_color='orange',
                        width=1))

fig.add_trace(go.Violin(x=['external test']*len(AUC_random),
                        y=AUC_random,
                        legendgroup='', scalegroup='', name='',
                        side='negative', spanmode='hard',
                        line_color='blue',
                        width=1))
fig.add_trace(go.Violin(x=['external test']*len(AUC_model),
                        y=AUC_model,
                        legendgroup='', scalegroup='', name='',
                        side='positive', spanmode='hard',
                        line_color='orange',
                        width=1))

fig.add_trace(go.Violin(x=['external train']*len(AUC_random),
                        y=external_train_random,
                        legendgroup='', scalegroup='', name='',
                        side='negative', spanmode='hard',
                        line_color='blue',
                        width=1))
fig.add_trace(go.Violin(x=['external train']*len(AUC_model),
                        y=external_train_model,
                        legendgroup='', scalegroup='', name='',
                        side='positive', spanmode='hard',
                        line_color='orange',
                        width=1))

# fig.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'Yes' ],
#                         y=df['total_bill'][ df['smoker'] == 'Yes' ],
#                         legendgroup='Yes', scalegroup='Yes', name='Yes',
#                         side='negative',
#                         line_color='blue')
#              )
# fig.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'No' ],
#                         y=df['total_bill'][ df['smoker'] == 'No' ],
#                         legendgroup='No', scalegroup='No', name='No',
#                         side='positive',
#                         line_color='orange')
#              )
fig.update_traces(meanline_visible=True)
fig.update_layout(title_text='none', violingap=0, violinmode='overlay',
                  height=800, width=1500,  # 设置图大小
                  font=dict(size=18),  # 字体大小
                  legend={'font': dict(size=25),  # legend字体大小
                          'itemsizing': 'constant', },  # legend图标大小
                  hoverlabel=dict(font=dict(size=20)))
# fig.show()
fig.write_image(r'C:\Users\HJ Wang\Desktop\result.pdf')
