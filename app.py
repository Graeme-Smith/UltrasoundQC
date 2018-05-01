# -*- coding: utf-8 -*-
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
from helpers  import *

# Import selected files. TODO Remove hard coded reference to import directory.
image_path = "/home/graeme/PycharmProjects/UltrasoundQC/test_data/"
image_files = import_batch_images(image_path)
user_selected_file = None


app = dash.Dash(__name__, static_folder='assets')
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True
app.config['suppress_callback_exceptions']=True

colors = {
    'background': '#111111',
    'text': '#ffffff'
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
                          dcc.Dropdown(
                              id='file-dropdown',
                              options=[{'label': i, 'value': i} for i in image_files],
                              # Populate drop down with file names
                              value=None
                          ),
                          dcc.Tabs(
                              tabs=[
                                  {'label': 'Select Image', 'value': 1},
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
    # Selected Image Tab
    if open_tab == 1:
        if file_name is None:
            pass
        else:
            encoded_image = base64.b64encode(open(file_name, 'rb').read())
            selected_image = 'data:image/png;base64,{}'.format(encoded_image)
            return html.Img(id='loaded_image', src=selected_image)
    # Image Analysis Tab
    if open_tab == 2:
        if file_name is None:
            pass
        else:
            pass
    # 3d Intensity Tab
    if open_tab == 3:
        if file_name is None:
            pass
        else:
            encoded_image = base64.b64encode(open(file_name, 'rb').read())
            pass

if __name__ == '__main__':
    app.run_server(debug=True,
                   port=8045)
