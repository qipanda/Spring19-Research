import pandas as pd
import numpy as np
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# Load data
sr_df = pd.read_csv("sr_df.csv")
sr_embeds = np.loadtxt(fname="sr_embeds.csv", delimiter=",")
p_embeds = np.loadtxt(fname="p_embeds.csv", delimiter=",")
with open('pred_map.pickle', 'rb') as handle:
    pred_map = pickle.load(handle)

# Calculate Pr(+|p, s, r, t) for each srt-p combination
srt_p_sig = 1.0/(1.0 + np.exp(-np.dot(sr_embeds, p_embeds.T)))
srt_p_sig_sorted = np.argsort(-srt_p_sig) # (argsort finds min to max, negative to do max to min)

# Calculate sr-dropdown options
options = [str(row["SOURCE"]) + "-" + str(row["RECEIVER"])
    for _, row in sr_df.loc[:, ["SOURCE", "RECEIVER"]].drop_duplicates().iterrows()]
options = [{"label":sr, "value":sr} for sr in options]

# Start the app
app = dash.Dash(__name__)
app.layout = html.Div(className="row", children=[
    html.Div(className="nine columns", children=[
        dcc.Graph(
            id='primary-graph',
            figure=go.Figure(
                data=[],
                layout=go.Layout(
                    # width=1500,
                    # height=800,
                    yaxis=dict(
                        range=[0.0, 1.0],
                        tick0=0.0,
                        dtick=0.25,
                        showgrid=True,
                        showticklabels=True,
                    ),
                )
            )
        )
    ]),
    html.Div(className="three columns", children=[
        dcc.Dropdown(
            id='sr-dropdown',
            options=options,
            value='ISR-PSE',
        ),
        dcc.Slider(
            id="top-slider",
            min=1,
            max=30,
            step=1,
            value=5,
            marks={i: str(i+1) for i in range(30)},
        ),
    ]),
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

@app.callback(
    Output("primary-graph", "figure"),
    [Input("sr-dropdown", "value"), Input("top-slider", "value")],
)
def update_figure(sr: str, num_top_preds: int) -> go.Figure :
    s, r = sr.split("-")
    df_slice = sr_df.loc[(sr_df["SOURCE"]==s) & (sr_df["RECEIVER"]==r), :]
    idxs = df_slice.index.values
    tracked_preds = np.unique(srt_p_sig_sorted[idxs, :num_top_preds].reshape(-1))
    values = srt_p_sig[idxs, :][:, tracked_preds]
    dates = [str(row["YEAR"]) + "-" + str(row["MONTH"]) for _, row in df_slice.iterrows()]

    data = []
    for pred_idx in tracked_preds:
        data.append(
            go.Scatter(
                x=dates,
                y=srt_p_sig[idxs, pred_idx],
                name=pred_map[pred_idx],
                mode="lines+markers",
            )
        )
        
    return go.Figure(
        data=data,
        layout=go.Layout(
            width=1500,
            height=800,
            yaxis=dict(
                range=[0.0, 1.0],
                tick0=0.0,
                dtick=0.25,
                showgrid=True,
                showticklabels=True
            ),
        )
    )

if __name__ == '__main__':
    app.run_server(debug=True)
