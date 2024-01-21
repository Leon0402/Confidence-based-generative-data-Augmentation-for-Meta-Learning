import plotly.graph_objects as go
import plotly.subplots
import numpy as np

import cdmetadl.dataset


def show_images_grid_plotly(data_set: cdmetadl.dataset.SetData):
    spacing = 100
    total_width = (128 + spacing) * data_set.max_number_of_shots - spacing
    total_height = (128 + spacing) * data_set.number_of_ways - spacing

    fig = plotly.subplots.make_subplots(
        rows=data_set.max_number_of_shots,
        cols=data_set.number_of_ways,
        subplot_titles=[f'Class {name}' for name in data_set.class_names],
        vertical_spacing=spacing / total_height,
        horizontal_spacing=spacing / total_width,
    )

    img_index = 0
    for class_index, num_shots in enumerate(data_set.number_of_shots_per_class):
        for shot_index in range(num_shots):
            img = data_set.images[img_index].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            fig.add_trace(go.Image(z=img), row=shot_index + 1, col=class_index + 1)
            img_index += 1

    fig.update_layout(height=total_height, width=total_width, showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig
