# Frond-End file

# required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# dash library
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
import plotly.express as px
from plotly.subplots import make_subplots


# input parameters list
input_params = html.Div([
    dbc.InputGroup([dbc.InputGroupText("input 1"), dbc.Input(
        placeholder="input1")], className="mb-3"),
    dbc.InputGroup([dbc.InputGroupText("input 2"), dbc.Input(
        placeholder="input2")], className="mb-3"),
    dbc.InputGroup([dbc.InputGroupText("input 3"), dbc.Input(
        placeholder="input3")], className="mb-3"),
    dbc.InputGroup([dbc.InputGroupText("input 4"), dbc.Input(
        placeholder="input4")], className="mb-3"),
    dbc.InputGroup([dbc.InputGroupText("input 5"), dbc.Input(
        placeholder="input5")], className="mb-3")
])


# initialize header buttons

# load buttons
load_button = dbc.Button("Load", id="load-button", className="me-2")
# save buttons
save_button = dbc.Button("Save", id="save-button", className="me-2")
# help buttons
help_button = dbc.Button("Help", id="help-button", className="me-2")
# edit buttons
edit_button = dbc.Button("Edit", id="edit-button", className="me-2")

# parameters buttons
collapse = html.Div([
    dbc.Button("Parameters", id="collapse-button",
               className="mb-3", n_clicks=0),
    dbc.Collapse(input_params, id="collapse", is_open=False)
])


# Abstract buttons
header_buttons = html.Div(dbc.Row([
    dbc.Col(
        html.Div(children=[load_button, save_button, edit_button, help_button])),
    dbc.Col(html.Div(children=collapse))
]))


def plot_traces(trace):

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        print_grid=True, vertical_spacing=0.05)

    fig.append_trace(px.line(y=trace)["data"][0], row=1, col=1)
    fig.append_trace(px.line(y=trace)["data"][0], row=2, col=1)
    fig.append_trace(px.line(y=trace)["data"][0], row=3, col=1)

    fig.update_layout(margin=dict(l=100, r=100, t=1, b=1),
                      width=900, height=400)

    return fig


# get accuracy plot
def get_acc_plot(train, val):

    # this will change with real data
    df = pd.DataFrame(
        {"Train": np.exp(-1/np.arange(2, 10, .1)), "Validation":  np.exp(-1/np.arange(2, 10, .1)) + .05})

    # start plotting
    fig = px.line(df, y=["Train", "Validation"])
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))

    return fig


# get confusion matrix
def get_confusion_mat():

    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    class_names = ['Type1', 'Type2', 'Type3']
    df = pd.DataFrame(np.round(cm, 3), columns=class_names, index=class_names)

    return df


# start tha main app
# Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like.
# The second part describes the interactivity of the application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# Graph div
trace_graphs = html.Div(children=[
    dcc.Graph(id="ch", figure=plot_traces(
        np.random.rand((1000))), responsive=True)
])

# lower Row (contains all learning graphs and informations + spectrums and histograms)
# confusion matrix table
# creating df which mimics sklearn output
table = dbc.Table.from_dataframe(
    get_confusion_mat(), striped=False, bordered=False, hover=True, index=True, responsive=True
)

# accuracy graph
acc_graph = dcc.Graph(id="accuracy", figure=get_acc_plot(
    train=[], val=[]), responsive=True, style={"width": "50vh", "height": "30vh"})


# lower row left-side
lower_row_left = dbc.Row([
    dbc.Col(table, width=4, align="center"),
    dbc.Col(acc_graph, width=4, align="center"),
    dbc.Col(dbc.Card("Train Info"), width=3)
], style={"display": "flex"})

# lower row right-side
lower_row_right = dbc.Row([
    dbc.Col(table, width=3),
    dbc.Col(dbc.Card("Hist"), width=3),
    dbc.Col(dbc.Card("Train Info"), width=3)
])

# lower row
lower_row = html.Div(children=[
    html.H3("Back-End informations", style={"border": "2px solid powderblue"}),
    dbc.Row([
        dbc.Col(lower_row_left),
        dbc.Col(lower_row_right)
    ], className="g-0")
])

# define app layout using dbc container
app.layout = dbc.Container(
    [html.H1("SleeZy v1.0.0"), header_buttons, trace_graphs, lower_row], fluid=True)

# all callBacks


@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
