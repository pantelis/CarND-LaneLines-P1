import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from addict import Dict
import helpers

# Parameters are a nested dictionary (addict library)
parameters = Dict()

parameters.Blurring.kernel_size = 5

parameters.Canny.low_threshold = 50
parameters.Canny.high_threshold = 150

parameters.Masking.vertices=np.array([[0,0],[0,0],[0,0],[0,0]], dtype=np.int32)

parameters.Hough.rho = 1 # distance resolution in pixels of the Hough grid
parameters.Hough.theta = np.pi/180 # angular resolution in radians of the Hough grid
parameters.Hough.threshold = 20 # minimum number of votes (intersections in Hough grid cell)
parameters.Hough.min_line_length = 2 # minimum number of pixels making up a line
parameters.Hough.max_line_gap = 10 # maximum gap in pixels between connectable line segments'


imageSourceDir = "test_images/"
imageTestDir = "test_images/"

for i in os.listdir(imageSourceDir):

    # Make copies into the test_images directory
    image = mpimg.imread(os.path.join(imageSourceDir, i))

    # Pull out the x and y sizes and make a copy of the image
    ysize = image.shape[0]
    xsize = image.shape[1]
    region_select = np.copy(image)

    # Convert to Grayscale
    image_gray=helpers.grayscale(image)

    # Blurring
    blurred_image = helpers.gaussian_blur(image_gray, parameters.Blurring.kernel_size)

    # Canny Transform
    edges = helpers.canny(blurred_image, parameters.Canny.low_threshold, parameters.Canny.high_threshold)

    # Four sided polygon to mask
    imshape = image.shape
    lower_left = (50, imshape[0])
    upper_left = (400, 320)
    upper_right = (524, 320)
    lower_right = (916, imshape[0])
    parameters.Masking.vertices = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    # masking
    masked_edges = helpers.region_of_interest(edges, parameters.Masking.vertices)

    # Run Hough on edge detected image
    hough_lines,hough_image = helpers.hough_lines(masked_edges, parameters.Hough.rho, parameters.Hough.theta, parameters.Hough.threshold,
                             parameters.Hough.min_line_length, parameters.Hough.max_line_gap)

    # classify left and right lane lines
    left_lane_lines, right_lane_lines = helpers.classify_left_right_lanes(hough_lines)

    # RANSAC fit left and right lane lines
    fitted_left_lane_points = helpers.ransac_fit_hough_lines(left_lane_lines)
    fitted_right_lane_points = helpers.ransac_fit_hough_lines(right_lane_lines)
    helpers.draw_model(image, fitted_left_lane_points, color=[255, 0, 0], thickness=2)
    helpers.draw_model(image, fitted_right_lane_points, color=[255, 0, 0], thickness=2)

    # Raw hough_lines image
    # helpers.draw_lines(raw_lines_img, hough_lines, color=[255, 0, 0], thickness=2)

    # 1D Interpolator - does not work as good as RANSAC so its commented out
    # interpolated_left_lane_points = helpers.interpolate_hough_lines(left_lane_lines)
    # interpolated_right_lane_points = helpers.interpolate_hough_lines(left_lane_lines)
    # helpers.draw_model(image, interpolated_left_lane_points, color=[105, 0, 0], thickness=2)
    # helpers.draw_model(image, interpolated_right_lane_points, color=[255, 0, 0], thickness=2)

    #superposed_image = helpers.weighted_img(hough_image, image, α=0.8, β=1., λ=0.)
    # then save them to the test_images directory.

    output_path_filename = os.path.join(imageSourceDir, "out_"+i)
    cv2.imwrite(output_path_filename,image)