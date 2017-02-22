import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from addict import Dict
import helpers

# Parameters are a nested dictionary (addict library)
parameters = Dict()

parameters.Blurring.kernel_size=5

parameters.Canny.low_threshold=50
parameters.Canny.high_threshold=150

parameters.Masking.vertices=np.array([[0,0],[0,0],[0,0],[0,0]], dtype=np.int32)

parameters.Hough.rho=1 # distance resolution in pixels of the Hough grid
parameters.Hough.theta=np.pi/180 # angular resolution in radians of the Hough grid
parameters.Hough.threshold=20 # minimum number of votes (intersections in Hough grid cell)
parameters.Hough.min_line_length=2 # minimum number of pixels making up a line
parameters.Hough.max_line_gap=10 # maximum gap in pixels between connectable line segments'


imageSourceDir = "test_images/"
imageTestDir = "test_images/"

#params = helpers.DictQuery(parameter_dict[0])

for i in os.listdir(imageSourceDir):

    # Make copies into the test_images directory
    image = mpimg.imread(os.path.join(imageSourceDir, i))

    # print('This image is: ', type(image),
    #       'with dimesions:', image.shape)
    #
    # # display original image
    # plt.figure("Original Image")
    # plt.imshow(image)
    # plt.show(block=False)

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
    upper_right = (524, 302)
    lower_right = (916, imshape[0])
    parameters.Masking.vertices = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)

    # masking
    masked_edges = helpers.region_of_interest(edges, parameters.Masking.vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on

    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, parameters.Hough.rho, parameters.Hough.theta, parameters.Hough.threshold, np.array([]),
                            parameters.Hough.min_line_length, parameters.Hough.max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
    plt.imshow(lines_edges)
    plt.show()

    # then save them to the test_images directory.
    output_path_filename = os.path.join(imageSourceDir, "out_"+i)
    cv2.imwrite(output_path_filename,lines_edges)