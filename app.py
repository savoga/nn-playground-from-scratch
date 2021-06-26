import dash
import dash_html_components as html
import dash_bootstrap_components as dbc

import pandas as pd
import dash_core_components as dcc
from dash.exceptions import PreventUpdate

import plotly.express as px

import utils as ut
import model as md

global df_data
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

            dbc.Button(
                    "Run neural network",
                    id="button-nn",
                    color="primary",
                    block=True,
                    className="mb-3",
                    ),

            dcc.Store(id='intermediate-value')
            # TODO: IT SEEMS DATAFRAMES CANNOT BE STORED

            ]
        )

@app.callback(
     [dash.dependencies.Output('intermediate-value', 'data'),
      dash.dependencies.Output('graph-data', 'figure')],
     [dash.dependencies.Input('button-generate', 'n_clicks'),
       dash.dependencies.Input('button-nn', 'n_clicks'),
       dash.dependencies.Input('intermediate-value','data')],
     [dash.dependencies.State('radio-data', 'value')])
def generateData(button_generate, button_nn, df_data_stored, radio_pattern):
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
        return df_data, fig_data
    elif button_id=='button-nn':
        print('NN button pressed')
        df_data = df_data_stored.copy()
        nn = md.NeuralNetwork(learning_rate=1.2, nodes_hidden=3)
        y_iterations = nn.train(df_data[['x','y']], df_data['cluster'], 1000)
        df_data['cluster'] = y_iterations[len(y_iterations)-1]
        fig_data = px.scatter(df_data, x="x", y="y", color='cluster')
        fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig_data.update_layout(coloraxis_showscale=False)
        return None, fig_data

# @app.callback(
#      dash.dependencies.Output('graph-data', 'figure'),
#      [dash.dependencies.Input('button-nn', 'value')])
# def runNeuralNetwork(fig_data):
#     nn = md.NeuralNetwork(learning_rate=1.2, nodes_hidden=3)
#     y_iterations = nn.train(df_data[['x','y']], df_data['cluster'], 1000)
#     df_data['cluster'] = y_iterations[len(y_iterations)-1]
#     fig_data = px.scatter(df_data, x="x", y="y", color='cluster')
#     fig_data.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
#     fig_data.update_layout(coloraxis_showscale=False)
#     return fig_data

if __name__ == '__main__':
    app.run_server(debug=True)