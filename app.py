# -*- coding: utf-8 -*-
import base64  # pybase64 required to serve static files to dash server.
import dash
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
static_image_route = '/static'  # Required to serve local images to Flask server

app = dash.Dash(__name__, static_folder='assets')
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

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

                          html.Img(id='loaded_image'),
                          html.Div(id='output-container'),
                          html.Div(
                              [
                                  html.Div(
                                      [
                                          dcc.Graph(id='main_graph')
                                      ],
                                      className='eight columns',
                                      style={'margin-top': '20'}
                                  ),
                                  html.Div(
                                      [
                                          dcc.Graph(id='vertical_graph')
                                      ],
                                      className='four columns',
                                      style={'margin-top': '20'}
                                  ),
                              ],
                              className='row'),
                          html.Div(
                              [
                                  html.Div(
                                      [
                                          dcc.Graph(id='horizontal_graph')
                                      ],
                                      className='eight columns',
                                      style={'margin-top': '20'}
                                  ),
                                  html.Div(
                                      [
                                          dcc.Graph(id='thumb_nail')
                                      ],
                                      className='four columns',
                                      style={'margin-top': '20'}
                                  ),
                              ],
                              className='row'),
                          html.Div(
                              [
                                  html.Div(
                                      [
                                          dcc.Graph(id='3d_graph')
                                      ],
                                      className='eight columns',
                                      style={'margin-top': '20'}
                                  ),
                                  html.Div(
                                      [
                                          dcc.Graph(id='test_graph3')
                                      ],
                                      className='four columns',
                                      style={'margin-top': '20'}
                                  ),
                              ],
                              className='row')

                      ])


@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('file-dropdown', 'value')])
def update_output(value):
    if value is not None:
        img = cv2.imread(value)
        print img
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('main_graph', 'figure'),
    [dash.dependencies.Input('file-dropdown', 'value')])
def update_plots(value):
    if value is not None:
        img, grey_image, blurred_image, threshold_image = import_image(value)
        ultrasound_cnt, cnt, convex, corners = detect_reverb(threshold_image)
        crop_img = mask_background(img, ultrasound_cnt)
        # Convert from numpy array to image format:
        crop_img = Image.fromarray(crop_img)
        horizontal_intensity = np.mean(crop_img, axis=0)  # Average by column
        vertical_intensity = np.mean(crop_img, axis=1)  # Average by row

        horiz_trace = go.Scatter(
            x=range(0, len(horizontal_intensity), 1),
            y=horizontal_intensity,
            mode='lines',
            name='lines'
        )

        int_data = [horiz_trace]
        layout = go.Layout(
            images=[dict(
                source=value,
                xref="x",
                yref="y",
                x=100,
                y=60,
                sizex=100,
                sizey=100,
                opacity=1,
                layer="below")],
            xaxis=dict(
                domain=[0, 0.45]
            ),
            yaxis=dict(
                domain=[0, 0.45]
            )
        )
        # fig = go.Figure(data=int_data, layout=layout)
        # py.plot(fig, filename='horizontal.html')
        my_figure = dict(data=int_data, layout=layout)
        return my_figure
    else:
        pass


@app.callback(
    dash.dependencies.Output('loaded_image', 'src'),
    [dash.dependencies.Input('file-dropdown', 'value')])
def update_image_src(value):
    encoded_image = base64.b64encode(open(value, 'rb').read())
    #app.layout = html.Div([
        #html.Img(src='data:image/png;base64,{}'.format(encoded_image))
    selected_image = 'data:image/png;base64,{}'.format(encoded_image)
    return selected_image


if __name__ == '__main__':
    app.run_server(debug=True)


