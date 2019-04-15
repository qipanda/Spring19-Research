# Import from Research directory
import sys, os
import itertools
sys.path.append(os.path.dirname(sys.path[0]))

# Import custom models
from Models.models import SRCTSoftmaxClassifier, SRCTSoftmaxModel
from Preprocessing.FullContextProcessor import FullContextProcessor

import pandas as pd
import numpy as np
import scipy as sp
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# Load data and model
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-month-indexed.txt", "\t")
fcp.createTwoWayMap("SOURCE", False)
fcp.createTwoWayMap("RECEIVER", False)
fcp.createTwoWayMap("PRED", False)

model = SRCTSoftmaxModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
                          r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                          p_cnt=len(fcp.df["PRED_IDX"].unique()),
                          T=len(fcp.df["TIME"].unique()),
                          K_s=150,
                          K_r=150,
                          K_p=300,)

model.load_state_dict(torch.load(
    "month_softmax_K300_lr1.00E+00_lam0.00E+00_alpha1.00E-03_bs32_epochs30.pt",
    map_location="cpu"))

s_embeds = model.s_embeds.weight.detach().numpy()
r_embeds = model.r_embeds.weight.detach().numpy()
p_embeds = model.p_embeds.weight.detach().numpy()

# calculate dates in x axis
dates = fcp.df.sort_values(by="TIME").loc[:, ["YEAR", "MONTH"]].drop_duplicates().values.tolist()
dates = [str(y) + "-" + str(m) for y, m in dates]

# Calculate sr-dropdown options
sr_combinations = itertools.product(
    fcp.twoway_maps["SOURCE"]["col_to_idx"].keys(),
    fcp.twoway_maps["RECEIVER"]["col_to_idx"].keys())
options = [{"label":s + "-" + r, "value":s + "-" + r} for s, r in sr_combinations if s != r]

# Start the app
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1("SR Top Pred Embeddings by Model Probability"),
    html.Div(children=[
        html.Div(style={"width":"70%", "float":"left"}, children=[
            dcc.Graph(
                id='primary-graph',
                figure=go.Figure(
                    data=[],
                    layout=go.Layout(
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
        html.Div(style={"width":"30%", "float":"left"}, children=[
            dcc.Dropdown(
                id='sr-dropdown',
                options=options,
                value="ISR-PSE",
                clearable=False,
            ),
            dcc.Slider(
                id="top-slider",
                min=1,
                max=10,
                step=1,
                value=1,
                marks={i: str(i) for i in range(1, 10+1)},
            ),
        ]),
    ]),
])

@app.callback(
    Output("primary-graph", "figure"),
    [Input("sr-dropdown", "value"), Input("top-slider", "value")],
)
def update_figure(sr: str, num_top_preds: int) -> go.Figure :
    # Get the s,r idxs
    s, r = sr.split("-")
    s = fcp.twoway_maps["SOURCE"]["col_to_idx"][s]
    r = fcp.twoway_maps["RECEIVER"]["col_to_idx"][r]

    # get p scores for all srt combinations
    X = torch.tensor([[s, r, t] for t in range(model.T)])
    p_probs = sp.special.softmax(model(X).detach().numpy(), axis=1)
                       
    p_probs_sorted = np.argsort(-p_probs) # min to max, neg to do max to min

    # find which p's to track based on num_top_preds
    tracked_preds = np.unique(p_probs_sorted[:, :num_top_preds].reshape(-1))

    data = []
    for pred_idx in tracked_preds:
        data.append(
            go.Scatter(
                x=dates,
                y=p_probs[:, pred_idx],
                name=fcp.twoway_maps["PRED"]["idx_to_col"][pred_idx],
                mode="lines",
            )
        )
        
    return go.Figure(
        data=data,
        layout=go.Layout(
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
