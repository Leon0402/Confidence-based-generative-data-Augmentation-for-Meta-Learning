import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Load CSV data
df = pd.read_csv('scripts.csv')

# Initialize Dash app
app = dash.Dash(__name__)

# Custom CSS styling for the table
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/litera/bootstrap.min.css']

# Define layout of the dashboard
app.layout = html.Div([
    html.H1("Scripts Dashboard", style={'textAlign': 'center'}),
    dcc.Interval(
        id='interval-component',
        interval=10*1000,  # in milliseconds
        n_intervals=0
    ),
    html.Div(id='live-update-text'),
    html.Div([
        dcc.Graph(id='table-graph')
    ], style={'width': '100%', 'display': 'inline-block', 'padding': '20px'})
], style={'font-family': 'Arial, sans-serif'})

# Define callback to update the table
@app.callback(
    Output('table-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_table(n):
    # Load updated data from CSV
    updated_df = pd.read_csv('scripts.csv')
    
    # Create table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(updated_df.columns),
                    fill_color='lightgrey',
                    align='left'),
        cells=dict(values=[updated_df[col] for col in updated_df.columns],
                   fill_color='white',
                   align='left'))
    ])
    
    fig.update_layout(title_text="Scripts Table")
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)