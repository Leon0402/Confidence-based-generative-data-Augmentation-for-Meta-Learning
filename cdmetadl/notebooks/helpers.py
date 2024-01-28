import plotly.graph_objects as go
import plotly.subplots
import numpy as np
import os
import plotly.io as pio
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
import cdmetadl.dataset
from plotly.subplots import make_subplots

def show_images_grid_plotly(data_set: cdmetadl.dataset.SetData, show_image=False):
    spacing = 100
    total_width = (128 + spacing) * data_set.number_of_ways - spacing
    total_height = (128 + spacing) * data_set.max_number_of_shots - spacing

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
    
    if show_image:
        fig.update_layout(title='Original data')
        file_path = "./grid.png"
        pio.write_image(fig, file_path)
        display(Image.open(file_path))
        os.remove(file_path)


    return fig

def create_plot(augmentor, task, conf_scores):
    augmented_set_generative = augmentor.augment(task.support_set, conf_scores=conf_scores)


    fig = show_images_grid_plotly(task.support_set)
    fig.update_layout(title='Original data')
    file_path = "./figname_support_set.png"
    pio.write_image(fig, file_path)
    display(Image.open(file_path))
    os.remove(file_path)


    fig = show_images_grid_plotly(augmented_set_generative)
    fig.update_layout(title='Generative Augmented data')
    file_path = "./figname_augmented_data.png"
    pio.write_image(fig, file_path)
    display(Image.open(file_path))
    os.remove(file_path)
    

def generate_edge_map_plot(augmentor, task, title):
    augmentor.augment(task.support_set, conf_scores=[0.1, 0.1, 0.1, 0.1, 0.1])

    spacing = 100
    total_width = (128 + spacing) * 3 - spacing
    total_height = (128 + spacing) * len(augmentor.generated_images)  - spacing
    
    # Create subplots with Plotly
    fig = make_subplots(rows=len(augmentor.generated_images), cols=3,
                        subplot_titles=['Original Image', 'Feature Map', 'Generated Image'])
    
    for i, data in enumerate(augmentor.generated_images):
        # Add traces to the corresponding subplot
        fig.add_trace(go.Image(z=np.array(data['original_image'])), row=i+1, col=1)
        fig.add_trace(go.Image(z=np.array(data['feature_map'])), row=i+1, col=2)
        fig.add_trace(go.Image(z=np.array(data['generated_image'])), row=i+1, col=3)
        fig.update_xaxes(title_text='X', row=i+1, col=1)
        fig.update_yaxes(title_text='Y', row=i+1, col=1)

        fig.update_xaxes(title_text='X', row=i+1, col=2)
        fig.update_yaxes(title_text='Y', row=i+1, col=2)

        fig.update_xaxes(title_text='X', row=i+1, col=3)
        fig.update_yaxes(title_text='Y', row=i+1, col=3)
    # Update layout for the entire figure
    fig.update_layout(title_text=title, width=total_width, height=total_height, showlegend=False)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    # Show the plot
    file_path = "./edge_map.png"
    pio.write_image(fig, file_path)
    display(Image.open(file_path))
    os.remove(file_path)
