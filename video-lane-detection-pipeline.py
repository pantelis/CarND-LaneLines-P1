# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
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
parameters.Hough.theta=np.pi/180 # angular resolution in radians of the Hough grid
parameters.Hough.threshold = 20 # minimum number of votes (intersections in Hough grid cell)
parameters.Hough.min_line_length = 2 # minimum number of pixels making up a line
parameters.Hough.max_line_gap = 10 # maximum gap in pixels between connectable line segments'

def process_image(image):

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
    hough_lines,raw_hough_lines_img = helpers.hough_lines(masked_edges, parameters.Hough.rho, parameters.Hough.theta, parameters.Hough.threshold,
                             parameters.Hough.min_line_length, parameters.Hough.max_line_gap)

    # classify left and right lane lines
    left_lane_lines, right_lane_lines = helpers.classify_left_right_lanes(hough_lines)

    # Raw hough_lines image
    helpers.draw_lines(raw_hough_lines_img, hough_lines, color=[255, 0, 0], thickness=2)

    # RANSAC fit left and right lane lines
    fitted_left_lane_points = helpers.ransac_fit_hough_lines(left_lane_lines)
    fitted_right_lane_points = helpers.ransac_fit_hough_lines(right_lane_lines)
    helpers.draw_model(image, fitted_left_lane_points, color=[255, 0, 0], thickness=2)
    helpers.draw_model(image, fitted_right_lane_points, color=[255, 0, 0], thickness=2)

    # 1D Interpolator - does not work as good as RANSAC so its commented out
    # interpolated_left_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # interpolated_right_lane_line = helpers.interpolate_hough_lines(left_lane_lines)
    # helpers.draw_model(image, interpolated_left_lane_line, color=[255, 0, 0], thickness=2)
    # helpers.draw_model(image, interpolated_right_lane_line, color=[255, 0, 0], thickness=2)

    # superpose images
    # superposed_image = helpers.weighted_img(image, raw_hough_lines_img, α=0.8, β=1., λ=0.)

    return image

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

# Now for the one with the solid yellow lane on the left. This one's more tricky!
yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)


# challenge_output = 'extra_raw.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)
