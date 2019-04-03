# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

# Load data
sr_df = pd.read_csv("sr_df.csv")
p_df = pd.read_csv("p_df.csv")

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    dcc.Graph(
        id='primary-graph',
        figure=go.Figure(
            data=[
                go.Scatter(
                    x = p_df["X"].values,
                    y = p_df["Y"].values,
                    mode = "markers",
                    name="Predicate-Embeddings"
                ),
                go.Scatter(
                    x = sr_df["X"].values,
                    y = sr_df["Y"].values,
                    mode = "markers",
                    name="Source-Receiver-Embeddings"
                )
            ],
            layout=go.Layout(
                width=700,
                height=500,
            )
        )
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
