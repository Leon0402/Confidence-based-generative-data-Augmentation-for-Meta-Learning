import pathlib

import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

all_metrics = ['Normalized Accuracy', 'Accuracy', 'Macro F1 Score', 'Macro Precision', 'Macro Recall']

dfs = {path.name: pd.read_pickle(path / "eval/eval.pkl") for path in pathlib.Path("./eval_output").iterdir()}
for key in dfs:
    dfs[key].insert(0, 'Model', key)


def sort_metrics(selected_metrics):
    return [metric for metric in all_metrics if metric in selected_metrics]


def get_data(models):
    return pd.concat([dfs[model_name] for model_name in models])


def convert_df_to_dash(df):
    """
    Converts a pandas data frame to a format accepted by dash
    Returns columns and data in the format dash requires
    """
    ids = ["".join([col for col in multi_col if col]) for multi_col in list(df.columns)]

    cols = [{"name": list(col), "id": id_} for col, id_ in zip(list(df.columns), ids)]
    data = [{k: v for k, v in zip(ids, row)} for row in df.values]

    return cols, data


app = dash.Dash(__name__)

app.layout = dmc.Container([
    dmc.AppShell(
        navbar=dmc.Navbar(
            width={"base": 200}, fixed=True, p="md", children=[
                dmc.Stack(
                    children=[
                        dmc.CheckboxGroup(
                            id='model-selector',
                            label="Select Models:",
                            value=list(dfs.keys()),
                            orientation="vertical",
                            children=[dmc.Checkbox(label=model, value=model) for model in dfs.keys()],
                        ),
                        dmc.CheckboxGroup(
                            id='metric-selector',
                            label="Select Metrics:",
                            value=all_metrics,
                            orientation="vertical",
                            children=[dmc.Checkbox(label=metric, value=metric) for metric in all_metrics],
                        ),
                    ]
                ),
            ]
        ), children=[
            dmc.Stack(
                children=[
                    dmc.Title("Model Averages", order=1),
                    dash.dash_table.DataTable(
                        id='average_table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                    ),
                    dmc.Title("Overall Frequency Histograms", order=1),
                    dash.html.Div(
                        id='histogram-container', style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(600px, 1fr))",
                        }
                    ),
                    dmc.Title("Scores per Feature", order=1),
                    dmc.SegmentedControl(
                        id="feature-selector",
                        value="dataset",
                        data=[
                            {
                                "value": "dataset",
                                "label": "Dataset"
                            },
                            {
                                "value": "way",
                                "label": "Way"
                            },
                            {
                                "value": "shot",
                                "label": "Shot"
                            },
                        ],
                        mt=10,
                    ),
                    dash.dash_table.DataTable(
                        id='scores_per_dataset_table',
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center'},
                        merge_duplicate_headers=True,
                    ),
                    dash.html.Div(
                        id='box-plot-container', style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(auto-fit, minmax(600px, 1fr))",
                        }
                    ),
                ]
            ),
        ]
    )
], fluid=True)


@app.callback(
    dash.Output('average_table', 'data'),
    [dash.Input('metric-selector', 'value'),
     dash.Input('model-selector', 'value')]
)
def average_table(metrics, models):
    df = get_data(models)
    averages = df.groupby('Model')[sort_metrics(metrics)].mean().reset_index()
    return averages.to_dict('records')


@app.callback(
    dash.Output('histogram-container', 'children'),
    [dash.Input('metric-selector', 'value'),
     dash.Input('model-selector', 'value')]
)
def overall_frequence_histogram(metrics, models):
    df = get_data(models)
    histogram_figures = []
    for metric in sort_metrics(metrics):
        hist_data = [group for _, group in df.groupby('Model')[metric]]
        fig = ff.create_distplot(hist_data, models, bin_size=.05)
        fig.update_layout(
            title_text=f'Overall Frequency Histogram ({metric})', xaxis_title_text=f'Score ({metric})',
            yaxis_title_text='Density'
        )

        histogram_figures.append(dash.dcc.Graph(figure=fig))

    return histogram_figures


@app.callback([dash.Output('scores_per_dataset_table', 'columns'),
               dash.Output('scores_per_dataset_table', 'data')], [
                   dash.Input('feature-selector', 'value'),
                   dash.Input('metric-selector', 'value'),
                   dash.Input('model-selector', 'value')
               ])
def scores_per_dataset(feature, metrics, models):
    df = get_data(models)
    averages = df.pivot_table(index=feature, columns='Model', values=metrics).reset_index()
    return convert_df_to_dash(averages)


@app.callback(
    dash.Output('box-plot-container', 'children'), [
        dash.Input('feature-selector', 'value'),
        dash.Input('metric-selector', 'value'),
        dash.Input('model-selector', 'value')
    ]
)
def box_plots(feature, metrics, models):
    df = get_data(models)

    return [
        dash.dcc.Graph(
            figure=px.box(
                df[["Model", feature, metric]], x=feature, y=metric, color="Model",
                title=f"Box Plot of {metric} by Model and {feature}"
            )
        ) for metric in sort_metrics(metrics)
    ]


if __name__ == '__main__':
    app.run_server(debug=True)
