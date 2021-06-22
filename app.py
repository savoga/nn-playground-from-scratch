import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import dash_core_components as dcc

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
                    children="Neural network playgroung from scratch",
                    style={'textAlign': 'center'}
                    ),

            html.Hr(),

            dcc.Markdown('Data generation'),

            dcc.RadioItems(
                id='data-generation',
                options=[
                    {'label': 'Blobs', 'value': 'blobs'},
                    {'label': 'Circles', 'value': 'circles'},
                    {'label': 'Moons', 'value': 'moons'}
                    ],
                value='blobs',
                labelStyle={'display': 'block'}
                    ),

            html.Hr(),

            dcc.Markdown('Visualization'),

            html.Div([
                        dcc.Graph(id="graph-data", figure=fig_data)
                    ]),

            dbc.Button(
                    "Run neural network",
                    id="button-nn",
                    color="primary",
                    block=True,
                    className="mb-3",
                    ),

            ]
        )

@app.callback(
     dash.dependencies.Output('graph-data', 'figure'),
     [dash.dependencies.Input('data-generation', 'value')])
def generateData(data_pattern):
    df_data = ut.generateData(data_pattern=data_pattern)
    df_data['cluster_start'] = 1 # at the beginning, all points belong to the same class
    fig_data = px.scatter(df_data, x="x", y="y", color='cluster_start')
    fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig_data.update_layout(coloraxis_showscale=False)
    return fig_data

'''
WARNING: impossible to have 2 callbacks updating the same figure
Option 1: use CallbackGrouper? (couldn't find how to use it...)
Option 2: use the same callback for both functionalities
'''
@app.callback(
     dash.dependencies.Output('graph-data', 'figure'),
     [dash.dependencies.Input('button-nn', 'value')])
def runNeuralNetwork(fig_data):
    nn = md.NeuralNetwork(learning_rate=1.2, nodes_hidden=3)
    y_iterations = nn.train(df_data[['x','y']], df_data['cluster'], 1000)
    df_data['cluster'] = y_iterations[len(y_iterations)-1]
    fig_data = px.scatter(df_data, x="x", y="y", color='cluster')
    fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig_data.update_layout(coloraxis_showscale=False)
    return fig_data

if __name__ == '__main__':
    app.run_server(debug=True)