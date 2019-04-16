# Import from Research directory
import sys, os
import itertools
from typing import List
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
from dash.dependencies import Input, Output, State

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
    "month_softmax_K300_lr1.00E+00_lam0.00E+00_alpha5.00E-02_bs32_epochs50.pt",
    map_location="cpu"))

s_embeds = model.s_embeds.weight.detach().numpy()
r_embeds = model.r_embeds.weight.detach().numpy()
p_embeds = model.p_embeds.weight.detach().numpy()

freqs = fcp.df.groupby(["SOURCE_IDX", "RECEIVER_IDX", "PRED_IDX", "TIME"])\
    .size()\
    .unstack(fill_value=0)\
    .stack()

# calculate dates in x axis
dates = fcp.df.sort_values(by="TIME").loc[:, ["YEAR", "MONTH"]].drop_duplicates().values.tolist()
dates = [str(y) + "-" + str(m) for y, m in dates]

# dropdown options
s_options = [{"label":s, "value":s} for s in fcp.twoway_maps["SOURCE"]["col_to_idx"].keys()]
r_options = [{"label":r, "value":r} for r in fcp.twoway_maps["RECEIVER"]["col_to_idx"].keys()]
p_options = [{"label":p, "value":p} for p in fcp.twoway_maps["PRED"]["col_to_idx"].keys()]

# Start the app
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1("Source-Receiver Skip-Gram Predicate Path Probabilities over Time"),
    html.Div(children=[
        html.Div(style={"width":"70%", "float":"left"}, children=[
            html.H2("Model graph"),
            dcc.Graph(
                id='primary-graph',
                figure=go.Figure(
                    data=[],
                    layout=go.Layout(
                        showlegend=True,
                        yaxis=dict(
                            showgrid=True,
                            showticklabels=True,
                        ),
                    )
                )
            ),
            html.H2("Frequency graph"),
            dcc.Graph(
                id="freq-graph",
                figure=go.Figure(
                    data=[],
                    layout=go.Layout(
                        showlegend=True,
                        yaxis=dict(
                            showgrid=True,
                            showticklabels=True,
                        ),
                    )
                )
            )
        ]),
        html.Div(style={"width":"30%", "float":"left"}, children=[
            html.H4("Source"),
            dcc.Dropdown(
                id='s-dropdown',
                placeholder="Select source country",
                options=s_options,
                value="ISR",
                clearable=False,
            ),
            html.H4("Receiver"),
            dcc.Dropdown(
                id="r-dropdown",
                placeholder="Select receiver country",
                options=r_options,
                value="PSE",
                clearable=False,
            ),
            html.H4("Number top predicates (on average over all timesteps)"),
            dcc.Slider(
                id="top-slider",
                min=1,
                max=20,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 20+1)},
            ),
            html.H4(""),
            html.Button("Plot top predicates",
                id="top-button",
                n_clicks_timestamp=1,
            ),
            html.H4("Number randomized predicates"),
            dcc.Slider(
                id="rand-slider",
                min=1,
                max=20,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 20+1)},
            ),
            html.H4(""),
            html.Button("Plot random predicates",
                id="rand-button",
                n_clicks_timestamp=0,
            ),
            html.H4("Manual predicate selection"),
            dcc.Dropdown(
                id="p-dropdown",
                placeholder="Select predicates to see",
                options=p_options,
                multi=True,
            ),
            html.H4(""),
            html.Button("Plot manual predicates",
                id="manual-button",
                n_clicks_timestamp=0,
            ),
        ]),
    ]),
])

@app.callback(
    [Output("primary-graph", "figure"),
     Output("freq-graph", "figure")],
    [Input("top-button", "n_clicks_timestamp"),
     Input("rand-button", "n_clicks_timestamp"),
     Input("manual-button", "n_clicks_timestamp")],
    [State("s-dropdown", "value"), 
     State("r-dropdown", "value"),
     State("p-dropdown", "value"),
     State("top-slider", "value"),
     State("rand-slider", "value")])
def update_figure(top_ts: int, rand_ts: int, man_ts: int, s: str, r: str, 
                  man_p: List[str], num_top_preds: int, num_rand: int) -> go.Figure:
    # Get the s,r idxs
    s = fcp.twoway_maps["SOURCE"]["col_to_idx"][s]
    r = fcp.twoway_maps["RECEIVER"]["col_to_idx"][r]

    # get p scores for all srt combinations
    X = torch.tensor([[s, r, t] for t in range(model.T)])
    p_probs = sp.special.softmax(model(X).detach().numpy(), axis=1)
    
    # see which button was most recently pressed
    button_tms = {"top":top_ts, "rand":rand_ts, "man":man_ts}
    recent_button = max(button_tms.items(), key=(lambda dict_tup: dict_tup[1]))[0]

    # Case 1: Want top preds
    if recent_button == "top":  
        p_probs_sorted = np.argsort(-np.mean(p_probs, axis=0)) # min to max, neg to do max to min
        tracked_preds = p_probs_sorted[:num_top_preds]
    elif recent_button == "rand":
        tracked_preds = np.random.choice(a=p_probs.shape[1], size=num_rand, replace=False)
    elif recent_button == "man":
        tracked_preds = [fcp.twoway_maps["PRED"]["col_to_idx"][p] for p in man_p]

    prob_data = []
    freq_data = []
    for pred_idx in tracked_preds:
        prob_data.append(
            go.Scatter(
                x=dates,
                y=p_probs[:, pred_idx],
                name=fcp.twoway_maps["PRED"]["idx_to_col"][pred_idx],
                mode="lines",
            )
        )

        # find freq data for this pred
        try:
            y_freq = freqs.loc[s, r, pred_idx].values
        except:
            y_freq = np.zeros(model.T)

        freq_data.append(
            go.Scatter(
                x=dates,
                y=y_freq,
                name=fcp.twoway_maps["PRED"]["idx_to_col"][pred_idx],
                mode="lines"
            )
        )
        
    prob_figure =  go.Figure(
        data=prob_data,
        layout=go.Layout(
            showlegend=True,
            yaxis=dict(
                showgrid=True,
                showticklabels=True
            ),
        )
    )
    freq_figure =  go.Figure(
        data=freq_data,
        layout=go.Layout(
            showlegend=True,
            yaxis=dict(
                showgrid=True,
                showticklabels=True
            ),
        )
    )

    return prob_figure, freq_figure

if __name__ == '__main__':
    app.run_server(debug=True)
