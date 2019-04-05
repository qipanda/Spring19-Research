# Import from Research directory
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

# Import custom models
from Models.models import SRCTClassifier, SRCTModel
from Preprocessing.FullContextProcessor import FullContextProcessor

import pandas as pd
import numpy as np
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

# Load data and model
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-year-indexed.txt", "\t")
model = SRCTModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
                  r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                  p_cnt=len(fcp.df["PRED_IDX"].unique()),
                  T=len(fcp.df["TIME"].unique()),
                  K_s=100,
                  K_r=100,
                  K_p=200,)

model.load_state_dict(torch.load(
    "K200_lr1.00E+00_lam0.00E+00_alpha1.00E-04_bs32_epochs10.pt",
    map_location="cpu"))

s_embeds = model.s_embeds.weight.detach().numpy()
r_embeds = model.r_embeds.weight.detach().numpy()
p_embeds = model.p_embeds.weight.detach().numpy()

# Get uniq SRs and Ps that appeared in the training data
sr_uniq = fcp.df.loc[:,["SOURCE", "RECEIVER", "SOURCE_IDX", "RECEIVER_IDX",
    "TIME", "YEAR"]].drop_duplicates().sort_values(by="YEAR").reset_index(drop=True)
sr_uniq.loc[:, "SOURCE_IDX"] += sr_uniq.loc[:, "TIME"]*model.s_cnt
sr_uniq.loc[:, "RECEIVER_IDX"] += sr_uniq.loc[:, "TIME"]*model.r_cnt
pred_map = fcp.df.loc[:, ["PRED_IDX", "PRED"]].drop_duplicates() \
                                              .set_index("PRED_IDX") \
                                              .to_dict()["PRED"]

# Calculate Pr(+|p, s, r, t) for each srt-p combination and sort them
srt_embeds = np.concatenate((s_embeds[sr_uniq["SOURCE_IDX"].values],
                            r_embeds[sr_uniq["RECEIVER_IDX"].values]), axis=1)
srt_p_sig = 1.0/(1.0 + np.exp(-np.dot(srt_embeds, p_embeds.T)))
srt_p_sig_sorted = np.argsort(-srt_p_sig) # (argsort finds min to max, negative to do max to min)

# Calculate sr-dropdown options
options = [{"label":row["SOURCE"] + "-" + row["RECEIVER"],
            "value":row["SOURCE"] + "-" + row["RECEIVER"]} 
            for _, row in sr_uniq.loc[:, ["SOURCE", "RECEIVER"]].drop_duplicates().iterrows()] 

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
                max=30,
                step=1,
                value=5,
                marks={i: str(i) for i in range(1, 30+1)},
            ),
        ]),
    ]),
])

@app.callback(
    Output("primary-graph", "figure"),
    [Input("sr-dropdown", "value"), Input("top-slider", "value")],
)
def update_figure(sr: str, num_top_preds: int) -> go.Figure :
    s, r = sr.split("-")
    df_slice = sr_uniq.loc[(sr_uniq["SOURCE"]==s) & (sr_uniq["RECEIVER"]==r), :]
    dates = [str(row["YEAR"]) for _, row in df_slice.iterrows()]
    idxs = df_slice.index.values # idx refrences rows in srt_embeds
    tracked_preds = np.unique(srt_p_sig_sorted[idxs, :num_top_preds].reshape(-1))
    values = srt_p_sig[idxs, :][:, tracked_preds]

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
