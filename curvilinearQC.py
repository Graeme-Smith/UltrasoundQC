from __future__ import print_function
import argparse
import pydicom as dicom
import os
import imutils
from scipy.misc import toimage
import plotly.offline as py
import plotly.graph_objs as go
# import pandas as pd
from helpers import *


""" 
Import Arguments from command line
"""
parser = argparse.ArgumentParser(
    description='Extract QC data from images produced by curvilinear ultrasound probes reverbing in air')
parser.add_argument('-file', '-f',
                    nargs='+',
                    help='Import single or multiple ultrasound images',
                    required=False)
parser.add_argument('-input_dir', '-i',
                    type=str,
                    help='File path to directory containing input images',
                    required=False)
parser.add_argument('-output_dir', '-o',
                    type=str,
                    help='File path to output results',
                    required=False)
args = parser.parse_args()

# User specified imaging files - List of strings as multiple image files may be provided.
# image_files = args.file
print(args.input_dir)
image_files = import_batch_images(args.input_dir)
print(image_files)
# User specified output directory for results/logs to be saved to. Directory will be created if it does not exist.
output_path = args.output_dir

# Create output directory in user specified directory for saving results:
output_directory = create_output_directory(output_path)

'''Import chosen image specified on the  command line or run on batch of images from a folder'''


img = cv2.imread(image_files[1])

# img = cv2.imread('/home/graeme/Desktop/reverb_images/20180220105417156.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/20180220105300406.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/20180220105459890.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/20180220105527234.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/20180220105659468.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20180219_1_83.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20131212_1_32.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20131212_1_20.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20131212_1_17.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20131212_1_16.jpg')
# img = cv2.imread('/home/graeme/Desktop/reverb_images/IMG_20131212_1_1.jpg')

'''If image captured from mobile phone perform perspective correction'''

'''Convert imported image to grey scale, apply blur, and threshold in preparation for feature detection'''

grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(grey_image, (15, 15), 0)
threshold_image = cv2.threshold(blurred_image, 25, 255, cv2.THRESH_BINARY)[1]

'''Feature detection in image'''
# find contours in the thresholded image
cnts = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

'''From the features returned from the image select the reverb pattern'''
cnt = select_contour(cnts)

# Calculate convex of selected contour:
convex = cv2.convexHull(cnt)

# Find the four corners of an ultrasound arc:
corners = corners_of_arc(convex)
# bottom_left, bottom_right, top_left, top_right = cornersOfArc(convex)

# Detect the ultrasound scan from image:
ultrasound_cnt, cnt, convex = clean_data(cnt, convex)
# cv2.drawContours(img, [ultrasound_cnt], -1, (0, 0, 255), 2)

# Create the basic black image
mask = np.zeros(img.shape, np.uint8)
# Draw a white contour
cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)

# Apply the mask and display the result
maskedImg = cv2.bitwise_and(img, mask)
equalised_img = cv2.equalizeHist(cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY))

# Create bounding rectangle:
x, y, w, h = cv2.boundingRect(ultrasound_cnt)
# cv2.rectangle(equalised_img,(x,y),(x+w,y+h),250,2)
crop_img = equalised_img[y:y + h, x:x + w]

# Calculate column and row intensities for numpy array:
horizontal_intensity = np.mean(crop_img, axis=0)  # Average by column
vertical_intensity = np.mean(crop_img, axis=1)  # Average by row

edges = cv2.Canny(crop_img, 100, 200)

# toimage(img).show()
toimage(crop_img).show()

toimage(edges).show()

# Plot horizontal pixel intensity.
horiz_trace = go.Scatter(
    x=range(0, len(horizontal_intensity), 1),
    y=horizontal_intensity,
    mode='lines',
    name='lines'
)

int_data = [horiz_trace]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    )
)
py.plot(int_data, filename='horizontal.html')

# Plot vertical pixel intensity.
vert_trace = go.Scatter(
    x=range(0, len(vertical_intensity), 1),
    y=vertical_intensity,
    mode='lines',
    name='lines'
)

int_data = [vert_trace]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    )
)
py.plot(int_data, filename='vert.html')

# Plot 3d surface of ultrasound reverb
crop_img[crop_img == 0] = 1

data = [
    go.Surface(
        z=crop_img
    )
]

layout = go.Layout(
    title='Ultrasound Reverberation 3d Surface Plot ',
    autosize=True,
    width=1000,
    height=1000,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='reverb-3d-surface.html')

lm, lc, rm, rc = detect_sides(cnt, corners_of_arc(cnt))

scan_angle = reverb_angle(lm, rm)

# export BROWSER=google-chrome

# Defining functions that will intersect
left_line = lambda x: lm * x + lc
right_line = lambda x: rm * x + rc

# Defining range and getting solutions on intersection points
x = np.linspace(0, 45, 10000)
result = find_intersection(left_line, right_line, [0])

# Printing out results for x and y
print(result, right_line(result))

plt.plot(cnt[:, 0, 0], cnt[:, 0, 1])
# plt.plot([top_left[0], 307.7655977,top_right[0]], [top_left[1],-392.72189105,top_right[1]])
x = np.linspace(-800, 800, 400)  # 100 linearly spaced numbers
y = right_line(x)
plt.plot(x, y)
y = left_line(x)
plt.plot(x, y)
plt.show()

# dst = cv2.linearPolar(grey_image, (392.72189105, -307.7655977), 600, cv2.WARP_FILL_OUTLIERS)

color = [0]  # black border
# border widths; I set them all to 150
top, bottom, left, right = [0, 2000, 0, 2000]

img_with_border = cv2.copyMakeBorder(grey_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

dst = cv2.logPolar(img_with_border, (392.72189105, -307.7655977), 400, cv2.WARP_FILL_OUTLIERS)
# toimage(dst).show()

rows, cols = dst.shape
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv2.warpAffine(dst, M, (cols, rows))

dst = dst[353:468, 553:980]

toimage(dst).show()

horizontal_intensity, vertical_intensity = pixel_intensities(dst)

# Plot horizontal pixel intensity.
horiz_trace = go.Scatter(
    x=range(0, len(horizontal_intensity), 1),
    y=horizontal_intensity,
    mode='lines',
    name='lines'
)

int_data = [horiz_trace]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    )
)
py.plot(int_data, filename='horizontal.html')

# Plot vertical pixel intensity.
vert_trace = go.Scatter(
    x=range(0, len(vertical_intensity), 1),
    y=vertical_intensity,
    mode='lines',
    name='lines'
)

int_data = [vert_trace]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    )
)
py.plot(int_data, filename='vert.html')

dst1 = dst.copy()

dst1 = dst1[37:86, :]

'''
levels = [0.0, 0.2, 0.5, 0.9, 1.5, 2.5, 3.5]
contour = plt.contour(X, Y, Z, levels, colors='k')
plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
contour_filled = plt.contourf(X, Y, Z, levels)
plt.colorbar(contour_filled)
plt.title('Plot from level list')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.show()


from skimage.feature import hessian_matrix, hessian_matrix_eigvals
def detect_ridges(gray, sigma=3.0):
    hxx, hyy, hxy = hessian_matrix(gray, sigma)
    i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
    return i1, i2

th2 = cv2.adaptiveThreshold(a,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,2,2)
vis = cv2.addWeighted(a,0.5,b,0.5,0)

s = skimage.feature.shape_index(crop_img, sigma=0.1)
'''

# Plot 3d surface of ultrasound reverb


data = [
    go.Surface(
        z=dst
    )
]

layout = go.Layout(
    title='Ultrasound Reverberation 3d Surface Plot ',
    autosize=True,
    width=1000,
    height=1000,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='reverb-polar-transformed.html')
