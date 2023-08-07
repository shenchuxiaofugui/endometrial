import pandas as pd
from pathlib import Path
import plotly.express as px

def draw_image(root, store_path):
    df = pd.read_csv(root, index_col=0)
    zhengfu = []
    neiwai = []
    value = []
    for i,j in zip(df.index, df["Coef"]):
        if "dilation" in i:
            neiwai.append("peri")
        else:
            neiwai.append("intra")
        if j > 0:
            zhengfu.append("positive")
            value.append(j)
        else:
            zhengfu.append("negative")
            value.append(-j)

    df["sign"] = zhengfu
    df["position"] = neiwai
    df["weights"] = value

    fig = px.bar(df, x="weights", y="features", color="sign", color_discrete_sequence=["#EF4143", "#73BDD6"],
                 orientation="h", height=1200, width=1600, text="features")  # color="sign",
    fig.update_layout(# title={"text":"Features' Weights", "x":0.5,"xanchor":"center", "yanchor": "top"},
                       yaxis={'categoryorder': 'total descending', }, # "showticklabels":False
                        font = dict(size=30),  # 字体大小
                        showlegend=False,  # legend = {'font': dict(size=24) legend字体大小  # legend图标大小
    ) # , yaxis={'categoryorder': 'total descending'}
    fig.update_traces(width=[0.8]*len(df))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=50), title_font=dict(size=50))
    fig.update_yaxes(showgrid=False, showticklabels=False ,title=None)
    #fig.show()
    fig.write_image(store_path)



def draw_Sunburst(root):
    for i in ["DWI", "T1CE", "T2"]:
        coef = [j for j in Path(root+f"/{i}/merge").glob("*coef.csv")]
        new_df = pd.read_csv(coef[0])
        new_df["Coef"] = abs(new_df["Coef"])
        new_df["parent"] = [i]*len(new_df)
        if i == "DWI":
            df = new_df
        else:
            df = pd.concat([df, new_df])
    all_df = pd.read_csv(r"C:/Users/handsome/Desktop/瘤周/result/LNM/combine/merge/SVM_coef.csv", index_col=0)
    all_df["features"] = ["DWI", "T1CE", "T2"]
    all_df["parent"] = [""] * len(all_df)
    df = pd.concat([df, all_df])
    fig = px.sunburst(
        df,
        names='features',
        parents='parent',
        values='Coef',
    )
    fig.show()


def draw_pie(root):
    df = pd.read_csv(root)
    colors = ["#469eb4", "#87cfa4", "fee89a"]
    for i in range(len(df)):
        df.iloc[i, 0] = df.iloc[i, 0].replace("_prediction", "")
        df.iloc[i, 1] = round(df.iloc[i, 1], 3)
    fig = px.pie(df, values="Coef", names="modals")
    fig.update_traces(textinfo="label+value", textfont_size=25,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
    fig.update(layout_showlegend=False)
    fig.write_image("C:/Users/handsome/Desktop/瘤周/FIG/feature weight/LVSI.png")
    fig.show()

# for j in ["LVSI", "LNM"]:
#     root = f"C:/Users/handsome/Desktop/瘤周/result/{j}"
#     for i in ["DWI", "T1CE", "T2"]:
#         coef = [j for j in Path(root+f"/{i}/merge").glob("*coef.csv")]
#         draw_image(str(coef[0]), f"C:/Users/handsome/Desktop/瘤周/FIG/feature weight/{j} {i}.png")
#draw_bar(r"C:/Users/handsome/Desktop/瘤周/result/LNM/T1CE/merge/SVM_coef.csv", f"C:/Users/handsome/Desktop/瘤周/FIG/feature weight/LVSI T1CE.jpg")
draw_pie(r"C:/Users/handsome/Desktop/瘤周/result/LVSI/combine/merge/SVM_coef.csv")
#draw_Sunburst(root)