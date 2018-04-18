# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_core_components as dcc
from helpers import *

# Import selected files. TODO Remove hard coded reference to import directory.
files = import_batch_images("/home/graeme/PycharmProjects/UltrasoundQC/test_data/")

app = dash.Dash()
app.layout = html.Div([
    html.H1(children='Ultrasound Reverb Analysis'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': i, 'value': i} for i in files],  # Populate drop down with file names
        value=None
    ),
    html.Div(id='output-container')
])


@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('file-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)
