import cv2
import datetime
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import scipy
from scipy import optimize


def create_output_directory(directory):
    """Create directory to contain results"""
    # Add date stamp to file name
    today = datetime.date.today()
    today = today.strftime('%d%b%Y')
    directory_name = "Ultrasound_QC_"
    directory = os.path.join(directory, directory_name + today)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def import_batch_images(relevant_path):
    """Create list of all images in user supplied directory"""
    included_extensions = ['jpg', 'png', 'dcm']
    file_names = [fn for fn in os.listdir(relevant_path)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


""" Geometric Functions which identifies the top, bottom and sides of a curvilinear reverb pattern."""


def select_contour(contours):
    """Select the contour relating to the ultasound image."""
    if len(contours) != 0:
        # Select contour with biggest area TODO add match check
        # cnt = max(cnts, key = cv2.contourArea)
        reverb_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        return reverb_contour


def clean_data(contour, convexed_contour):
    """Cleanup the reverb pattern within the image."""
    # Calculate moments and centre points:
    m = cv2.moments(contour)
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])

    # The top curve is defined the best in cnts while the bottom curve is defined best
    # in convex
    bottom_left, bottom_right, top_left, top_right = corners = corners_of_arc(convexed_contour)
    # Isolate coordinates for top curve from cnt
    top_curve = isolate_curve(contour, corners, top=True)

    # Calculate position to insert top curve into convex:
    left_x, left_y = np.where(convexed_contour[:, 0] == top_left)
    right_x, right_y = np.where(convexed_contour[:, 0] == top_right)

    # Delete top straight line from convex coordinates: TODO
    temp = convexed_contour.copy()
    # Insert top curve where straight line was in convex coordinates: TODO Remove hardwired indexes
    temp = np.concatenate((temp[:left_x[2], 0], (top_curve[::-1])[:-1], temp[left_x[2]:, 0], temp[0]), axis=0)
    temp = np.concatenate((temp, temp[:0]), axis=0)
    temp = np.delete(temp, (len(top_curve) + left_x[0]), axis=0)
    ultrasound_contour = temp

    return ultrasound_contour, contour, convexed_contour


def detect_sides(contour, corners):
    """Detect sides from ultrasound image."""
    bottom_left, bottom_right, top_left, top_right = corners

    # Extract points composing left side
    temp = contour[np.where(contour[:, :, 0] < top_left[0])]
    left_side = temp[np.where(temp[:, 1] < bottom_left[1])]

    # Extract points composing right side
    temp = contour[np.where(contour[:, :, 0] > top_right[0])]
    right_side = temp[np.where(temp[:, 1] < bottom_right[1])]

    # Fit line to points
    temp = np.vstack([left_side[:, 0], np.ones(len(left_side[:, 0]))]).T
    lm, lc = np.linalg.lstsq(temp, left_side[:, 1], rcond=None)[0]

    temp = np.vstack([right_side[:, 0], np.ones(len(right_side[:, 0]))]).T
    rm, rc = np.linalg.lstsq(temp, right_side[:, 1], rcond=None)[0]

    return lm, lc, rm, rc


def corners_of_arc(coordinates):
    """Detects the corners of a ultrasound scan image."""
    # Calculate moments and centre point of contour:
    m = cv2.moments(coordinates)
    cX = int(m["m10"] / m["m00"])
    cY = int(m["m01"] / m["m00"])

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


""" Geometric Functions which identify characteristics of the curvilinear reverb pattern,
 for example focal point & angle of sides"""


def find_intersection(func1, func2, x0):
    """Find the intersection of two functions."""
    return scipy.optimize.fsolve(lambda x: func1(x) - func2(x), x0)


def reverb_angle(lm, rm):
    """Returns between the two sides of a reverb image"""
    l_deg = math.degrees(math.atan2(lm, 1))
    r_deg = math.degrees(math.atan2(rm, 1))
    deg = 180-(round(abs(l_deg), 0) + round(abs(r_deg), 0))
    return deg


def fit_curve(contour):
    """Fit polynomial curve to data points."""
    coefs = np.polynomial.polynomial.polyfit(x=contour[:, 0, 0], y=contour[:, 0, 1], deg=2)
    ffit = np.polynomial.polynomial.polyval(contour[:, 0, 0], coefs)
    return coefs, ffit


# plt.plot(cnt[:,0,0], ffit)
# plt.plot(cnt[:,0,0], cnt[:,0,1])


def isolate_curve(contour, corners=[], top=True):
    # Returns the coordinates relating to the top curve:
    bottom_left, bottom_right, top_left, top_right = corners
    if top:
        # Isolate coordinates for top curve from cnt
        temp = contour[np.where(contour[:, :, 0] >= top_left[0])]
        temp = temp[np.where(temp[:, 0] <= top_right[0])]
        temp = temp[np.where(temp[:, 1] <= bottom_left[1])]
        curve = temp[np.where(temp[:, 1] <= bottom_right[1])]
    # Returns the coordinates relating to the bottom curve:
    else:
        temp = contour[np.where(contour[:, :, 0] >= bottom_left[0])]
        temp = temp[np.where(temp[:, 0] <= bottom_right[0])]
        temp = temp[np.where(temp[:, 1] >= bottom_left[1])]
        curve = temp[np.where(temp[:, 1] >= bottom_right[1])]
    return curve


plt.gca().invert_yaxis()
# x = isolate_curve(cnt, False)
# plt.plot(x[:,0],x[:,1])
# x = isolate_curve(cnt)
# plt.plot(x[:,0],x[:,1])


"""Plotting Functions to visualise data"""


def pixel_intensities(image_array):
    """For a greyscale image/numpy array returns the average pixel intensity across each column and row."""
    horizontal_intensity = np.mean(image_array, axis=0)  # Average by column
    vertical_intensity = np.mean(image_array, axis=1)  # Average by row
    return horizontal_intensity, vertical_intensity


def plot3d(greyscale_image, log_transform=False):
    """Convert a greyscale image into a 3d surface"""
    if log_transform:
        #Plot data
        print "todo"
    else:
        #Log transform data before plotting
        print "todo"
    pass
