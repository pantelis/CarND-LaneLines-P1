# Finding Lane Lines on the Road

Pantelis Monogioudis

NOKIA

**Summary**

The goal of this project is to make a pipeline that finds lane lines on
the road from monocular video camera streams.

---

[//]: # (Image References)

[out_solidWhiteCurve]: ./test_images/out_solidWhiteCurve.jpg "Detecting
lanes in an image containing a solid white curve"

[out_solidWhiteRight]: ./test_images/out_solidWhiteRight.jpg "Detecting
lanes in an image containing a right solid white line"

[out_solidYellowCurve]: ./test_images/out_solidYellowCurve.jpg "Detecting
lanes in an image containing a solid yellow curve"

[out_solidYellowCurve2]: ./test_images/out_solidYellowCurve2.jpg "Detecting
lanes in an image containing another solid yellow curve"


---

## Pipeline
The pipeline consists of 5 stages.

1. First, the images are converted to grayscale using the corresponding
   openCV function.
2. Then noise is spatially averaged out using a symmetric Gaussian
   kernel. The standard deviation of the Gaussian kernel in the x and y
   direction was kept the same in the initial experimentation.
3. A Canny edge detection algorithm was then used to processed the
   blurred image.
4. A probabilistic Hough transform processed all Canny-detected edge
   pixels in the quadrilateral region of interest.
5. The identified edge lines were then classified as belonging to a
   right or left lane. The classification was based on a simple slope
   criterion.
6. The points belonging to the classified to each lane where fit using a
   RANSAC regressor and the produced linear model was superposed on the
   image.

## Performance
The figures below demonstrate the end result superposing the identified
lane lines on still images:

![Detecting
lanes in an image containing a solid white curve][out_solidWhiteCurve]

![Detecting
lanes in an image containing a right solid white line][out_solidWhiteRight]

![Detecting lanes in an image containing a solid
yellow curve][out_solidYellowCurve]

![Detecting lanes in an image containing another solid
yellow curve][out_solidYellowCurve2]

The developed simple pipeline was also tested in video scenes with both
white and yellow lines.

* Despite its simplicity, the pipeline performed well and was able to
  identify both white and yellow lanes.
* We noticed that in some specific scenes as other vehicles come close
  to the lanes, the RANSAC regressor will get confused and create
  spurious lines. This can be observed instantaneously on the yellow.mp4
  video. Effectively the outliers created by the other car can throw
  instantaneously the RANSAC regressor off.
* The pipeline, however, performed poorly on the extra.mp4 video scene
  that contains shadows from trees, nearby to the left lane concrete
  media barriers and other objects that are picked up by
  the Canny detector and are not filtered adequately by the Hough
  transform.


##Possible improvements
1. There is far more work that is required to have a successful outcome
   in the challenging extra.mp4 video. The work is particularly
   challenging as the project did not offer any training datasets where
   models can be trained on and parameter tuning was almost ad hoc
   (trial and error). One source of improvement is therefore the
   improvement of the test setup that will allow engineers to tune their
   algorithms.

2. Algorithmically, after a literature survey on the subject, we can
   summarise the following ideas that almost certainly lead to
   performance improvements: a. The Gaussian kernel applied during the
   blurring step can be asymmetric with the standard deviation in the
   y-direction larger than the x-direction. The reason is that we are
   trying to detect vertical lines whose length is larger than their
   width. b. The parameters of the RANSAC regressor can be tuned.
   Especially the parameter residual_threshold that can tune the maximum
   residual for a data sample to be classified as an inlier. This will
   probably eliminate the aforementioned in the performance section
   RANSAC issues with nearby objects. c. We can an Inverse Perspective
   Mapping (IPM) which generates a bird's eye view of the road. In this
   view, the nearby to the left yellow line objects or concrete medians
   can be filtered out much better by selecting an appropriate region of
   interest.

