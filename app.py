# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # Required as matplotlib must be run from main thread
from matplotlib import pyplot as plt

import base64  # pybase64 required to serve static files to dash server.
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import flask
from PIL import Image
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
from helpers import *
import StringIO as io


# Import selected files. TODO Remove hard coded reference to image directory.
image_path = "/home/graeme/PycharmProjects/UltrasoundQC/test_data/"
image_files = import_batch_images(image_path)
user_selected_file = None

app = dash.Dash(__name__, static_folder='assets')
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.config['suppress_callback_exceptions']=True

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

app.layout = html.Div(style={'backgroundColor': colors['background']},
                      children=[
                          html.Link(href='/assets/main.css', rel='stylesheet'),
                          html.H1(children='Ultrasound Reverb Analysis',
                                  style={
                                      'textAlign': 'left',
                                      'color': colors['text']
                                  }
                                  ),
                          html.Div(children='''
                          Select image file from drop down box below.
                          ''',
                          style = {
                              'textAlign': 'left',
                              'color': colors['text']
                          }
                                   ),
                          dcc.Dropdown(
                              id='file-dropdown',
                              options=[{'label': i, 'value': i} for i in image_files],
                              # Populate drop down with file names
                              value=None
                          ),
                          dcc.Tabs(
                              tabs=[
                                  {'label': 'Selected Image', 'value': 1},
                                  {'label': 'Image Analysis', 'value': 2},
                                  {'label': '3d Intensity Plot', 'value': 3}
                              ],
                              value=1,
                              id='tabs',
                              vertical=False
                          ),
                          html.Div([], id='tab-output')
                      ])


@app.callback(Output('tab-output', 'children'),
              [Input('tabs', 'value'), Input('file-dropdown', 'value')])
def display_content(open_tab, file_name):
    '''Import chosen image and display tab appropriate items'''

    if file_name is not None:
        # Import selected image and perform initial preprocessing:
        img, grey_image, blurred_image, threshold_image = import_image(file_name)
        # Detect reverb feature in image:
        ultrasound_cnt, cnt, convex, corners = detect_reverb(threshold_image)
        # Return cropped image
        crop_img = mask_background(img, ultrasound_cnt)
        dst = curvilinear_to_linear(grey_image)

    # Selected Image Tab
    if open_tab == 1:
        if file_name is None:
            pass
        else:
            fig = plt.figure()
            temp_img = cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
            plt.imshow(temp_img)

            imgdata = io.StringIO()
            fig.savefig(imgdata, format='png', bbox_inches='tight')

            encoded_image = base64.b64encode(imgdata.getvalue())
            selected_image = 'data:image/png;base64,{}'.format(encoded_image)
            return html.Div(
                [
                    html.Img(id='loaded_image', src=selected_image)
                ]
            )

    # Image Analysis Tab
    if open_tab == 2:
        if file_name is None:
            pass
        else:
            # Calculate column and row intensities for numpy array:
            horizontal_intensity, vertical_intensity = pixel_intensities(dst)
            img_width, img_height = crop_img.shape
            fig = plt.figure()
            plt.imshow(crop_img, aspect='equal', extent=None)
            plt.axis('off')

            imgdata = io.StringIO()
            fig.savefig(imgdata, format='png', bbox_inches='tight')

            encoded_image = base64.b64encode(imgdata.getvalue())
            selected_image = 'data:image/png;base64,{}'.format(encoded_image)

            fig = plt.figure()
            plt.imshow(dst, aspect='equal', extent=None)
            plt.axis('off')

            imgdata = io.StringIO()
            fig.savefig(imgdata, format='png', bbox_inches='tight')

            encoded_image = base64.b64encode(imgdata.getvalue())
            transformed_image = 'data:image/png;base64,{}'.format(encoded_image)

            return  html.Div(
        [

        html.Div(
        [
            html.Img(id='cropped_image',
                     src=selected_image,
                className='eight columns',
                style={'margin-top': '0'}
            )
        ],
        className='row'),
        html.Div(
            [
                html.Img(id='transformed_image',
                         src=transformed_image,
                         className='eight columns',
                         style={'margin-top': '0'}
                         ),
                html.Div(
                    [
                        dcc.Graph(
                            id='horizontal_intensity',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=horizontal_intensity,
                                        y=range(0, len(horizontal_intensity), 1),
                                        mode='lines',
                                        name='lines'
                                    )
                                ],
                                'layout': go.Layout(
                                    xaxis=dict(
                                        domain=[0, 0.45]
                                    ),
                                    yaxis=dict(
                                        domain=[0, 0.45],
                                        autorange='reversed'
                                    ),
                                    autosize=False,
                                    width=1000,
                                    height=1500,
                                    margin=dict(
                                        l=65,
                                        r=50,
                                        b=65,
                                        t=90
                                    )
                                )
                            }
                        )
                    ],
                    className='four columns',
                    style={'margin-top': '0'}
                )
            ],
            className='row'),
        html.Div(
            [
                html.Div(

                    [
                        dcc.Graph(
                            id='vertical_intensity',
                            figure={
                                'data': [
                                    go.Scatter(
                                        x=range(0, len(vertical_intensity), 1),
                                        y=vertical_intensity,
                                        mode='lines',
                                        name='lines'
                                    )
                                ],
                                'layout': go.Layout(
                                    xaxis=dict(
                                        domain=[0, 0.45]
                                    ),
                                    yaxis=dict(
                                        domain=[0, 0.45]
                                    ),
                                    autosize=False,
                                    width=1500,
                                    height=500,
                                    margin=dict(
                                        l=0,
                                        r=0,
                                        b=0,
                                        t=0
                                    ),
                                )
                            }
                        )
                    ],

                    className='eight columns',
                    style={'margin-top': '0'}
                )
            ],
            className='row')
                    ]
            )


    # 3d Intensity Tab
    if open_tab == 3:
        if file_name is None:
            pass
        else:
            # Plot 3d surface of ultrasound reverb
            crop_img[crop_img == 0] = 1 # TODO WTF?
            return html.Div([
            dcc.Graph(
                id='Ultrasound-Reverb-3d-Plot',
                figure={
                    'data': [
                        go.Surface(
                            z=crop_img
                        )
                    ],
                    'layout': go.Layout(
                        title='Ultrasound Reverberation 3d Surface Plot ',
                        autosize=True,
                        width=1000,
                        height=750,
                        margin=dict(
                            l=65,
                            r=50,
                            b=65,
                            t=90
                        )
                    )
                }
            )
            ])

if __name__ == '__main__':
    app.run_server(debug=True,
                   port=8047)