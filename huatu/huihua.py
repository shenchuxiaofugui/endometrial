import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


# features = ["T1CE_dilation_6_log-sigma-3-0-mm-3D_firstorder_Variance", "T1CE_dilation_6_log-sigma-3-0-mm-3D_firstorder_MeanAbsoluteDeviation", "T1CE_dilation_6_log-sigma-5-0-mm-3D_firstorder_10Percentile"]
# suoxie = ["LoG-3 Var", "LoG-3 MAD", "LoG-5 10%"]
features = ["T1CE_dilation_9_original_gldm_DependenceEntropy", "T1CE_dilation_9_log-sigma-1-0-mm-3D_firstorder_Variance", "T1CE_resampled.nii_log-sigma-5-0-mm-3D_glszm_HighGrayLevelZoneEmphasis"]
suoxie = ["Ori DE", "LoG-1 Var", "LoG-5 HGZE"]
#df = pd.read_excel(r"\\mega\syli\dataset\EC_all\model\train_features.xlsx")
df1 = pd.read_csv(r"C:\Users\handsome\Desktop\瘤周\result\LNM\T1CE\merge\selected_train_data.csv")
df2 = pd.read_csv(r"C:\Users\handsome\Desktop\瘤周\result\LNM\T1CE\merge\selected_test_data.csv")
df = pd.concat((df1, df2), axis=0)
# fig = px.violin(df, x="Category", y="T2", color="Category") #, x="features", y="values"
#fig = px.violin(df, x=list(df), y=list(df),color=list(df))
fig = go.Figure()
colors = ["blue", "orange"]
flag = True
for j, i in enumerate(features):
    if i == "label":
        continue
    fig.add_trace(go.Violin(
        x=[suoxie[j]]*len(df["label"] == 1),
        y=df[i][df["label"] == 1],
        legendgroup="positive",
        scalegroup="positive",
        name="positive",
        side="positive",
        line_color="lightseagreen",
        showlegend=flag,
        width=0.7,
        points=False
        # box_visible=True,
        # meanline_visible=True
    ))
    flag = False
flag = True
for j, i in enumerate(features):
    if i == "label":
        continue
    fig.add_trace(go.Violin(
        x=[suoxie[j]]*len(df["label"] == 1),
        y=df[i][df["label"] == 0],
        legendgroup="negative",
        scalegroup="negative",
        name="negative",
        side="negative",
        line_color="mediumpurple",
        showlegend=flag,
        width=0.7,
        points=False
        # box_visible=True,
        # meanline_visible=True    f"feature{j+1}"
    ))
    flag = False


fig.update_traces(
    width=0.7,
    selector=dict(type='violin'))
fig.update_layout(title_text='LNM Features', violingap=0.25, violinmode='overlay',
                  height=800, width=1600,  # 设置图大小
                  font=dict(size=18),  # 字体大小
                  legend={'font': dict(size=25),  # legend字体大小
                          'itemsizing': 'constant', },  # legend图标大小
                  hoverlabel=dict(font=dict(size=20)),
                )  #violingroupgap=0.8
fig.show()
