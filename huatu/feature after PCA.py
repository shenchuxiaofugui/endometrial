import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import numpy as np

N = 500
trace0 = go.Scatter(
    x=np.random.randn(N),
    y=np.random.randn(N) + 2,
    name='Above',
    mode='markers',
    marker=dict(
        size=10,
        color='rgba(152, 0, 0, 0.8)',
        line=dict(
            width=2,
            color='rgb(0, 0, 0)'
        )
    )
)

trace1 = go.Scatter(
    x=np.random.randn(N),
    y=np.random.randn(N) - 2,
    name='Below',
    mode='markers',
    marker=dict(
        size=10,
        color='rgba(255, 182, 193, 0.9)',
        line=dict(width=2,
                  )
    )
)

data = [trace0, trace1]
layout = dict(
    title='Styled Scatter',
    yaxis=dict(zeroline=False), xaxis=dict(zeroline=False))

plotly.offline.plot(data, layout, filename='./styled-scatter.html')