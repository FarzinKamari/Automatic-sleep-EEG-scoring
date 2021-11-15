# Frontend file

# required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb

# dash library
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
import plotly.express as px
from plotly.subplots import make_subplots

# dash extension
from dash_extensions import Keyboard

# university logo path
University_Logo = "https://upload.wikimedia.org/wikipedia/de/9/97/Eberhard_Karls_Universit%C3%A4t_T%C3%BCbingen.svg"

# search bar
'''search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(dbc.Button("Search epoch", color="primary", class_name="ms-2", n_clicks=0),
            width="auto",
        ),
    ],
    class_name="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)'''

# draft training parameters list
input_params = html.Div([
    dbc.InputGroup([dbc.InputGroupText("ML algorithm"), dbc.Input(
        placeholder="Which ML you want to use?")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("nThread"), dbc.Input(
        placeholder="The number of dedicated threads")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("GPU"), dbc.Input(
        placeholder="1 for yes, 0 for no")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("Lag time"), dbc.Input(
        placeholder="How much you can wait for training")], class_name="mb-1")
])

# config menu items
config_menu_items = html.Div(
    [
    dbc.DropdownMenuItem("Human", id="dropdown-menu-item-1"),
    dbc.DropdownMenuItem("Rodent", id="dropdown-menu-item-2"),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Custom", id="dropdown-menu-item-3"),
    ],
)

# import configs
input_config = html.Div([
    dbc.InputGroup([
        dbc.DropdownMenu(config_menu_items, label="Species"),
        dbc.Input(id="input-group-dropdown-input", placeholder="Epoch length")]),
    dbc.InputGroup([dbc.InputGroupText("Channels to import"), dbc.Input(
        placeholder="How many channels do you have?")])
])


# advanced parameters button
param_collapse = html.Div([
    dbc.Button("Advanced parameters", id="param_collapse_button", n_clicks=0),
    dbc.Collapse(input_params, id="param_collapse", is_open=False)
])


# navigation toolbar with logo, software title and a button

navbar = dbc.NavbarSimple(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(html.A(html.Img(src=University_Logo, height="40px"), href="http://www.physiologie2.uni-tuebingen.de/"), width="auto"),
                dbc.Col(html.H1("Sleeezy", style={'color': '#003D7F', 'fontSize': 45})),


                dbc.Col(dbc.DropdownMenu(
                children=[
                dbc.DropdownMenuItem("Please specify the below inputs, and then press Load", header=True),
                dbc.InputGroup([
                    dbc.DropdownMenu(config_menu_items, label="Species"),
                       dbc.Input(id="input-group-dropdown-input", placeholder="Epoch length"),
                dbc.InputGroup([dbc.InputGroupText("Channels to import"), dbc.Input(
                        placeholder="Number of channels")])
                ]),
             
                dbc.Col(dbc.Button("Load", id="import_collapse_button2", n_clicks=0), width="auto"),
                
                ],
                nav=False,
                in_navbar=True,
                label="Import",
                ), width = "auto",
                ),

                dbc.Col(html.Div(
                 [
                    dbc.Button("Loading params", id="open-offcanvas", n_clicks=0),
                    dbc.Offcanvas(
                    html.P(
                        "Below are the channels of your dataset. "
                        "Please select which ones you want to load, "
                        "and then, press the Load button. "
                        "This can take a couple of minutes!"
                    ),
                    id="offcanvas",
                    title="Loading...",
                    is_open=False,
                        ),
                    ]
                ), width="auto"),

                
                #dbc.Col(dbc.Collapse(input_config, id="import_collapse", is_open=False), width="auto"),
                dbc.Col(dbc.Button("Save", id="save-button"), width="auto"),
                #dbc.Col(dbc.Button("Edit", id="edit-button"), width="auto"),
                dbc.Col(dbc.Button("Advanced", id="param_collapse_button", n_clicks=0), width="auto"),
                dbc.Col(dbc.Collapse(input_params, id="param_collapse", is_open=False), width="auto"),
                dbc.Col(dbc.Button("About Us", id="about-us-button"), width="auto"),
                dbc.Col(dbc.Button("Help", id="help-button"), width="auto"),
                dbc.Col(dbc.NavbarToggler(id="navbar-toggler", n_clicks=0), width="auto"),
                dbc.Col(dbc.Collapse(dbc.Input(type="search", placeholder="Search"), id="navbar-collapse", is_open=True), width="auto"),
                dbc.Col(dbc.Button("Search epoch", n_clicks=0), width="auto")
            ],
        align="center",
        justify ="center",
        ),
    fluid=False,
    ),
    
    links_left = True,
    sticky="top",
    #class_name="d-flex align-items-evenly",
    color="info",
    dark=False,
    fluid=False,
)


def plot_traces(index):

    # call trace from inside
    trace = np.random.rand((1000)) + index
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        print_grid=False, vertical_spacing=0.05)

    fig.add_trace(px.line(y=trace)["data"][0], row=1, col=1)
    fig.add_trace(px.line(y=trace)["data"][0], row=2, col=1)
    fig.add_trace(px.line(y=trace)["data"][0], row=3, col=1)

    fig.update_layout(margin=dict(l=100, r=100, t=1, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      width=900, height=400,
                      )
    return fig


# get accuracy plot
def get_acc_plot():

    ### call any function to receive train, val accuracy values inside this function #####

    # this will change with real data
    df = pd.DataFrame(
        {"Train": np.exp(-1/np.arange(2, 10, .1)), "Validation":  np.exp(-1/np.arange(2, 10, .1)) + .05})

    # start plotting
    fig = px.line(df, y=["Train", "Validation"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=1, r=1, t=1, b=1),
                      legend=dict(
                          yanchor="bottom",
                          y=0.1,
                          xanchor="left",
                          x=0.45),
                      xaxis={"showgrid": False, "showline": True},
                      yaxis={"showgrid": False, "showline": True}
                      )

    return fig


# spectrum & histograms
def get_hists():

    # call trace from inside
    trace = np.random.rand((100))

    fig = make_subplots(rows=2, cols=3, shared_yaxes=True,
                        print_grid=False, vertical_spacing=0.25, horizontal_spacing=0.05,
                        subplot_titles=("Power Spectrums", "", "", "Amplitude Histograms", "", ""))

    fig.add_trace(px.line(y=trace)["data"][0], row=1, col=1)
    fig.add_trace(px.histogram(x=trace)["data"][0], row=2, col=1)
    fig.add_trace(px.line(y=trace)["data"][0], row=1, col=2)
    fig.add_trace(px.histogram(x=trace)["data"][0], row=2, col=2)
    fig.add_trace(px.line(y=trace)["data"][0], row=1, col=3)
    fig.add_trace(px.histogram(x=trace)["data"][0], row=2, col=3)

    #fig.update_yaxes(title_text="Power Spectrums", row=1, col=1)
    #fig.update_yaxes(title_text="Amplitude Histograms", row=2, col=1)

    fig.update_layout(margin=dict(l=1, r=1, t=20, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)'
                      )

    return fig


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])


# Graph div
trace_graphs = html.Div(children=[
    dcc.Graph(id="ch", responsive=True)
])

# lower Row (contains all learning graphs and informations + spectrums and histograms)
# confusion matrix table
# creating df which mimics sklearn output
table = dbc.Table.from_dataframe(
    get_confusion_mat(), striped=False, bordered=False, hover=True, index=True, responsive=True, size="sm"
)

# accuracy graph
acc_graph = dcc.Graph(id="accuracy", figure=get_acc_plot(
), responsive=True, style={"width": "40vh", "height": "30vh"})


# hist graphs
hist_graph = dcc.Graph(id="hist-graphs", figure=get_hists(),
                       responsive=True, style={"width": "90vh", "height": "30vh"})

# lower row left-side
lower_row_left = dbc.Container(dbc.Row([
    dbc.Col(table, width=3, align="center"),
    dbc.Col(acc_graph, width=5, align="center"),
    dbc.Col(dbc.Card("Train Info"), width=4)
], style={"display": "flex"}))

# lower row right-side
lower_row_right = dbc.Container(dbc.Row([
    hist_graph
]))

# lower row
lower_row = dbc.NavbarSimple(html.Div(children=[
    dbc.Container(html.H3("Analytics",
            style={"border": "2px solid powderblue", "margin-bottom": "1em", "margin-top": "1em"}), fluid=True),
    dbc.Container(dbc.Row([
        dbc.Col(lower_row_left, width=7),
        dbc.Col(lower_row_right, width=5)
    ], class_name="g-5"))
]),
    fixed="bottom",
    )


# detecting keyboard keys
my_keyboard = html.Div(Keyboard(id="keyboard"))


# define app layout using dbc container
app.layout = dbc.Container(
    html.Div([my_keyboard, navbar, trace_graphs, lower_row]), fluid=True)

# all callBacks

@app.callback(
    Output("ch", "figure"),
    [Input("keyboard", "keydown"), Input("keyboard", "n_keydowns")]
)
def keydown(event, n_keydowns):
    print(n_keydowns)
    if n_keydowns:
        if event["key"] == "ArrowLeft":
            fig = plot_traces(index=np.random.randint(0, 10, 1))

        elif event["key"] == "ArrowRight":
            fig = plot_traces(index=np.random.randint(0, 10, 1))

        print(event["key"], n_keydowns)
        return fig
    return plot_traces(index=0)
    


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

'''@app.callback(
    Output("import_collapse", "is_open"),
    [Input("import_collapse_button", "n_clicks")],
    [State("import_collapse", "is_open")],
)
def toggle_import_collapse(n, is_open):
    if n:
        return not is_open
    return is_open'''

@app.callback(
    Output("input-group-dropdown-input", "value"),
    [
        Input("dropdown-menu-item-1", "n_clicks"),
        Input("dropdown-menu-item-2", "n_clicks"),
        Input("dropdown-menu-item-3", "n_clicks"),
    ],
)
def on_button_click(n1, n2, n3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "dropdown-menu-item-3":
        return "Enter your customized epoch length"
    elif button_id == "dropdown-menu-item-1":
        return "30"
    else:
        return "10"

@app.callback(
    Output("param_collapse", "is_open"),
    [Input("param_collapse_button", "n_clicks")],
    [State("param_collapse", "is_open")],
)
def toggle_param_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
