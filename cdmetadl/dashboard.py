import pathlib
import argparse
import time

import plotly.graph_objects as go
import dash
import dash_mantine_components as dmc
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px

all_metrics = ['Normalized Accuracy', 'Accuracy', 'Macro F1 Score', 'Macro Precision', 'Macro Recall']

parser = argparse.ArgumentParser(description='Dashboard')
parser.add_argument('--eval-output-path', type=pathlib.Path, required=True, help='Path to eval output')
args = parser.parse_args()

short_names = {"cross-domain": "CD", "within-domain": "WD", "domain-independent": "DI"}


def read_df(path: pathlib.Path) -> pd.DataFrame:
    df: pd.DataFrame = pd.read_pickle(path)
    path_parts = path.relative_to(args.eval_output_path).parts
    df.insert(0, 'Model', f"{path_parts[2]} ({short_names[path_parts[3]]})")

    scores_per_data = []

    if "generative_aug" in path_parts[1]:
        augmentation_type = "Generative Augmentation"
    elif "standard_aug" in path_parts[1]:
        augmentation_type = "Standard Augmentation"
    elif "pseudo_aug" in path_parts[1]:
        augmentation_type = "Pseudo Augmentation"
    elif "baseline_constant_confidence" in path_parts[1]:
        augmentation_type = "No Augmentation"
    else:
        print(path_parts[1])


    for idx, group in df.groupby("Dataset"):
        scores = {"Model": f"{path_parts[2]} ({short_names[path_parts[3]]})",
                  "Dataset": group["Dataset"].iloc[0],
                  "Augmentation Type": augmentation_type,
                  "Number of Ways": group["Number of Ways"].iloc[0],
                  "Number of Shots": group["Number of Shots"].iloc[0],
                  "Normalized Accuracy": group["Normalized Accuracy"].mean(),
                  "Accuracy": group["Accuracy"].mean(),
                  "Macro F1 Score": group["Macro F1 Score"].mean(),
                  "Macro Precision": group["Macro Precision"].mean(),
                  "Macro Recall": group["Macro Recall"].mean(),
                  }

        if scores["Number of Shots"] >= 11:
            scores["Number of Shots"] -= 10
        scores_per_data.append(scores)

    return pd.DataFrame(scores_per_data)


full_df = pd.concat([read_df(filepath) for filepath in args.eval_output_path.glob('**/evaluation.pkl')])

# Keep only datasets present in all models
# nb_models_used = full_df.groupby("Dataset")["Model"].nunique()
# keep_groups = nb_models_used == nb_models_used.max()
# full_df = full_df[full_df['Dataset'].isin(keep_groups[keep_groups].index)]

# full_df = full_df[full_df['Dataset'].isin(keep_groups[keep_groups].index)]
dfs = {model: full_df[full_df["Model"] == model] for model in full_df["Model"].unique()}

print("Loaded!")
print(dfs)

def convert_df_to_dash(df):
    """
    Converts a pandas data frame to a format accepted by dash
    Returns columns and data in the format dash requires
    """
    print("Test12")
    
    ids = ["".join([col for col in multi_col if col]) for multi_col in list(df.columns)]

    cols = [{"name": list(col), "id": id_} for col, id_ in zip(list(df.columns), ids)]
    data = [{k: v for k, v in zip(ids, row)} for row in df.values]

    return cols, data


app = dash.Dash(__name__)

app.layout = dmc.Container([
    dash.dcc.Store(id='model-store', storage_type='memory'),
    dash.dcc.Store(id='metric-store', storage_type='memory'),
    dmc.AppShell(
        navbar=dmc.Navbar(
            width={"base": 400}, fixed=True, p="md", children=[
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
                        dmc.CheckboxGroup(
                            id='dataset-selector',
                            label="Select Datasets:",
                            value=full_df['Dataset'].unique(),
                            orientation="horizontal",
                            children=[
                                dmc.Checkbox(label=dataset, value=dataset, style={"width": "100px"})
                                for dataset in full_df['Dataset'].unique()
                            ],
                        )
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
                    # dmc.Title("Overall Frequency Histograms", order=1),
                    # dash.html.Div(
                    #     id='histogram-container', style={
                    #         "display": "grid",
                    #         "gridTemplateColumns": "repeat(auto-fit, minmax(600px, 1fr))",
                    #     }
                    #),
                    # dmc.Title("Scores per Feature", order=1),
                    # dmc.SegmentedControl(
                    #     id="feature-selector",
                    #     value="Dataset",
                    #     data=[
                    #         {
                    #             "value": "Dataset",
                    #             "label": "Dataset"
                    #         },
                    #         {
                    #             "value": "Number of Ways",
                    #             "label": "Number of Ways"
                    #         },
                    #         {
                    #             "value": "Number of Shots",
                    #             "label": "Number of Shots"
                    #         },
                    #     ],
                    #     mt=10,
                    # ),
                    # dash.dash_table.DataTable(
                    #     id='scores_per_dataset_table',
                    #     style_table={'overflowX': 'auto'},
                    #     style_cell={'textAlign': 'center'},
                    #     merge_duplicate_headers=True,
                    # ),
                    # dash.html.Div(
                    #     id='box-plot-container', style={
                    #         "display": "grid",
                    #         "gridTemplateColumns": "repeat(auto-fit, minmax(600px, 1fr))",
                    #     }
                    # ),
                ]
            ),
        ]
    )
], fluid=True)


@app.callback(dash.Output('model-store', 'data'), [dash.Input('model-selector', 'value')])
def update_models(selected_models):
    print("Test1")
    df = pd.concat([dfs[model_name] for model_name in selected_models]).to_dict("records")
    print("Test1 finished")
    return df


@app.callback(dash.Output('metric-store', 'data'), [dash.Input('metric-selector', 'value')])
def update_metrics(selected_metrics):
    print("Test2")
    return [metric for metric in all_metrics if metric in selected_metrics]


def sort_metrics(selected_metrics):
    print("Test3")
    return [metric for metric in all_metrics if metric in selected_metrics]


@app.callback([dash.Output('average_table', 'data'),
               dash.Output('average_table', 'style_data_conditional')],
              [dash.Input('model-store', 'data'), dash.Input('metric-store', 'data')])
def average_table(model_store_data, metrics):
    print("TEST!!!!")
    df = pd.DataFrame(model_store_data)

    means = df.groupby('Model')[metrics].mean().round(decimals=3)
    print("MEANS!!!!!")
    print(means)
    std_devs = df.groupby('Model')[metrics].std().round(decimals=3)
    averages = means.astype(str) + " ± " + std_devs.astype(str)
    averages.reset_index(inplace=True)

    style = [{
        'if': {
            'filter_query': f'{{{metric}}} contains {means[metric].max()}',
            'column_id': metric
        },
        'fontWeight': 'bold'
    } for metric in metrics]

    return averages.to_dict('records'), style


# @app.callback(
#     dash.Output('histogram-container', 'children'),
#     [dash.Input('model-store', 'data'), dash.Input('metric-store', 'data')]
# )
# def overall_frequence_histogram(model_store_data, metrics):
#     df = pd.DataFrame(model_store_data)  #.groupby('Model')
#     # model_names = list(df.groups.keys())

#     histogram_figures = []
#     for metric in metrics:
#         fig = px.histogram(df, x=metric, color="Model")

#         # hist_data = [group for _, group in df[metric]]
#         # fig = ff.create_distplot(hist_data, model_names, bin_size=.05)
#         # fig.update_layout(
#         #     title_text=f'Overall Frequency Histogram ({metric})', xaxis_title_text=f'Score ({metric})',
#         #     yaxis_title_text='Density'
#         # )

#         histogram_figures.append(dash.dcc.Graph(figure=fig))

#     return histogram_figures


# @app.callback([dash.Output('scores_per_dataset_table', 'columns'),
#                dash.Output('scores_per_dataset_table', 'data')], [
#                    dash.Input('feature-selector', 'value'),
#                    dash.Input('metric-store', 'data'),
#                    dash.Input('model-store', 'data')
#                ])
# def scores_per_dataset(feature, metrics, model_store_data):
#     df = pd.DataFrame(model_store_data)
#     averages = df.pivot_table(index=feature, columns='Model', values=metrics).reset_index().round(decimals=3)
#     return convert_df_to_dash(averages)


# @app.callback(
#     dash.Output('box-plot-container', 'children'), [
#         dash.Input('feature-selector', 'value'),
#         dash.Input('metric-store', 'data'),
#         dash.Input('model-store', 'data'),
#     ]
# )
# def box_plots(feature, metrics, model_store_data):
#     df = pd.DataFrame(model_store_data)

#     return [
#         dash.dcc.Graph(
#             figure=px.box(
#                 df[["Model", feature, metric]], x=feature, y=metric, color="Model",
#                 title=f"Box Plot of {metric} by Model and {feature}"
#             )
#         ) for metric in metrics
#     ]


if __name__ == '__main__':
    app.run_server(debug=True)