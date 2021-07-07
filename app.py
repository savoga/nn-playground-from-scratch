import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

import dash_core_components as dcc
from dash.exceptions import PreventUpdate

import plotly.express as px

import utils as ut
import model as md

df_data = pd.DataFrame(columns=['x','y','cluster'])
fig_data = px.scatter(df_data, x="x", y="y")

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY])
server = app.server

app.layout = dbc.Container(
            [
            html.H1(
                    children="2-layer neural network playgroung from scratch",
                    style={'textAlign': 'center'}
                    ),

            html.Hr(),

            dcc.Markdown('Data generation'),

            dcc.RadioItems(
                id='radio-data',
                options=[
                    {'label': 'Blobs', 'value': 'blobs'},
                    {'label': 'Circles', 'value': 'circles'},
                    {'label': 'Moons', 'value': 'moons'}
                    ],
                value='blobs',
                labelStyle={'display': 'block'}
                    ),

            dbc.Button(
                    "Generate data",
                    id="button-generate",
                    color="primary",
                    block=True,
                    className="mb-3",
                    ),

            html.Hr(),

            dcc.Markdown('Visualization'),

            html.Div([
                        dcc.Graph(id="graph-data", figure=fig_data)
                    ]),

            dcc.Markdown('Hidden units'),

            dcc.RadioItems(
                id='radio-units',
                options=[
                    {'label': '1', 'value': 1},
                    {'label': '2', 'value': 2},
                    {'label': '3', 'value': 3},
                    {'label': '4', 'value': 4}
                    ],
                value=1,
                labelStyle={'display': 'block'}
                    ),

            dbc.Button(
                    "Run neural network",
                    id="button-nn",
                    color="primary",
                    block=True,
                    className="mb-3",
                    ),

            dcc.Store(id='intermediate-value')

            ]
        )

@app.callback(
     [dash.dependencies.Output('intermediate-value', 'data'),
      dash.dependencies.Output('graph-data', 'figure')],
     [dash.dependencies.Input('button-generate', 'n_clicks'),
       dash.dependencies.Input('button-nn', 'n_clicks'),
       dash.dependencies.Input('radio-units', 'value'),
       dash.dependencies.Input('intermediate-value','data')],
     [dash.dependencies.State('radio-data', 'value')])
def generateData(button_generate, button_nn, nb_hh, df_data_stored, radio_pattern):
    if button_generate is None:
        raise PreventUpdate
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id=='button-generate':
        df_data = ut.generateData(data_pattern=radio_pattern)
        df_data['cluster_start'] = 1 # at the beginning, all points belong to the same class
        fig_data = px.scatter(df_data, x="x", y="y", color='cluster_start')
        fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig_data.update_layout(coloraxis_showscale=False)
        return df_data.to_json(date_format='iso', orient='split'), fig_data
    elif button_id=='button-nn':
        df_data = pd.read_json(df_data_stored, orient='split')
        nn = md.NeuralNetwork(learning_rate=1.2, nodes_hidden=nb_hh)
        y_iterations = nn.train(np.array(df_data[['x','y']]), np.array(df_data['cluster']), 1000)
        df_data['cluster'] = y_iterations[len(y_iterations)-1]
        fig_data = px.scatter(df_data, x="x", y="y", color='cluster')
        fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig_data.update_layout(coloraxis_showscale=False)
        return df_data.to_json(date_format='iso', orient='split'), fig_data

if __name__ == '__main__':
    app.run_server(debug=True)