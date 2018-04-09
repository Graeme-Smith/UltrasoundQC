import numpy as np
from matplotlib import pyplot as plt
import scipy
import cv2
import math

''' Geometric Functions which identifies the top, bottom and sides of a curvilinear reverb pattern.'''

# Detect sides from ultrasound image
def detect_sides(contour, corners):
    bottom_left, bottom_right, top_left, top_right = corners

    # Extract points composing left side
    temp = contour[np.where(contour[:, :, 0] < top_left[0])]
    left_side = temp[np.where(temp[:, 1] < bottom_left[1])]

    # Extract points composing right side
    temp = contour[np.where(contour[:, :, 0] > top_right[0])]
    right_side = temp[np.where(temp[:, 1] < bottom_right[1])]

    # Fit line to points
    A = np.vstack([left_side[:,0], np.ones(len(left_side[:,0]))]).T
    lm, lc = np.linalg.lstsq(A, left_side[:,1],rcond=None)[0]

    A = np.vstack([right_side[:,0], np.ones(len(right_side[:,0]))]).T
    rm, rc = np.linalg.lstsq(A, right_side[:,1],rcond=None)[0]

    return lm, lc, rm, rc

# Detects the corners of a ultrasound scan image:
def cornersOfArc(coordinates):

    # Calculate moments and centre point of contour:
    M = cv2.moments(coordinates)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Bottom left & bottom right corners are simply the left most and right most points respectively.
    bottom_left = tuple(coordinates[coordinates[:, :, 0].argmin()][0])
    bottom_right = tuple(coordinates[coordinates[:, :, 0].argmax()][0])
    # Split coordinates between left and right of centre
    right = coordinates[np.where(coordinates[:, :, 0] > cX)]
    left = coordinates[np.where(coordinates[:, :, 0] < cX)]
    # Find highest point on each side:
    top_left = tuple(left[left[:, 1].argmin()])
    top_right = tuple(right[right[:, 1].argmin()])
    return bottom_left, bottom_right, top_left, top_right

''' Geometric Functions which identify characteristics of the curvilinear reverb pattern,
 for example focal point & angle of sides'''

# Function to find the intersection of two functions

def findIntersection(func1, func2, x0):
    return scipy.optimize.fsolve(lambda x : func1(x) - func2(x), x0)

# Returns between the two sides of a reverb image:
def reverb_angle(lm, rm):
    l_deg = math.degrees(math.atan2(lm, 1))
    r_deg = math.degrees(math.atan2(rm, 1))
    deg = 180-(round(abs(l_deg),0) + round(abs(r_deg),0))
    return deg

'''Plotting Functions to visualise data'''

# For a greyscale image/numpy array returns the average pixel intensity across each column and row.

def pixelIntensities(imageArray):
    horizontal_intensity = np.mean(imageArray, axis=0)  # Average by column
    vertical_intensity = np.mean(imageArray, axis=1)  # Average by row
    return horizontal_intensity, vertical_intensity

# Convert a greyscale image into a 3d surface
def plot3d(greyscale_image, log_transform=False):
    if log_transform:
        #Plot data
    else:
        #Log transform data before plotting
    return

