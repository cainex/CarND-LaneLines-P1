# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[solidWhiteCurve]: ./test_images/solidWhiteCurve.jpg
[solidWhiteRight]: ./test_images/solidWhiteRight.jpg
[solidYellowCurve]: ./test_images/solidYellowCurve.jpg
[solidYellowCurve2]: ./test_images/solidYellowCurve2.jpg
[solidYellowLeft]: ./test_images/solidYellowLeft.jpg
[whiteCarLaneSwitch]: ./test_images/whiteCarLaneSwitch.jpg

[solidWhiteCurveFound]: ./test_images_output/solidWhiteCurve.png
[solidWhiteRightFound]: ./test_images_output/solidWhiteRight.png
[solidYellowCurveFound]: ./test_images_output/solidYellowCurve.png
[solidYellowCurve2Found]: ./test_images_output/solidYellowCurve2.png
[solidYellowLeftFound]: ./test_images_output/solidYellowLeft.png
[whiteCarLaneSwitchFound]: ./test_images_output/whiteCarLaneSwitch.png

---

### Reflection

### 1. Pipeline description

The lane finding pipeline is comprised of 6 stages:

1. Convert input image to grayscale

   This first step is simply to convert the color image to grayscale for easier processing, using the provided grayscale() function. The resulting grayscale image is saved to 'gray_img' varaible.

2. Apply guassian filter to smooth and remove noise

   This stage is done to remove any high-frequency noise from the image to get better results from the Canny edge detection. This uses the provided gaussian_blur() function. This is tunable with the p2_guassian_kernel_size variable. A setting of 5 was used. The results are saved into gauss_img.

3. Perform Canny edge detection on image

   In this stage, all edges are detected with Canny edge detection using the provided canny() function. This stage is tunable using p3_canny_low_threshold and p3_canny_high_threshold. Values of 50 and 200 were used respectively. The edge detected image output is saved to canny_img

4. Mask a region of interest

   This stage creates a polygon to mask off uninteresting areas of the Canny edge detection results. Since we are detecting lane lines, we have a rough idea of where we expect those lines will appear in the image. Masking off the region will remove noise from other edges detected that we are not interesting. The bounds are a tunable set of variables prepended with "p4" to signify pipe-stage 4, and "x" or "y" to denote the axis. For the x axis, we have a top-left bound "tlb", bottom-left bound "blb", top-right bound "trb" and and bottom-right bound "brb". Since the polygon will have horizontal lines at the top and bottom, we only need two y bounds "bb" for bottom-bound and "tb" for top-bound. 

   Since we cannot predetermine the image size, and we want the detection to work on any image shape, the bounds are set to be relative to the image size. The top x axis bounds are to the +/- 10% of the center of the image. The bottom x axis bounds are set to be 10% of the image size in from the edge of the image. The top y bound is set to the 60% mark of the image and the bottom y bound is set to the 90% mark of the image. There is a part of the image at the bottom of the image masked off to allow for the camera to be recording part of the front of the car (as was the case in the challenge section).

   With the bounds defined, a set of vertices is created to define the poygon:

   * (p4_x_blb, p4_y_bb)
 
   * (p4_x_tlb, p4_y_tb)
 
   * (p4_x_trb, p4_y_tb)
 
   * (p4_x_brb, p4_y_bb)

   These vertices are added to a numpy array and used with the provided region_of_interest() function. The resulting masked image is saved to roi_canny_img.

5. Apply Hough Transform

   This pipestage is fairly complex, and occurs in two parts:

   1. Generate Hough Lines

      Hough lines are created from the masked Canny image from the previous pipestage using the provided hough_lines() function. This function is tunable from the following parameters:

        * p5_rho = 1;
        * p5_theta = np.pi/270
        * p5_threshold = 3
        * p5_min_line_len = 8
        * p5_max_line_gap = 3

   2. Apply draw_lines()

      The hough_lines() function internally calls draw_lines() to create an image with the Hough lines drawn. This function was heavily modified to perform several stages of filtering to create stable and connected lane lines.

      1. Sort and filter Hough lines by slope

         Initially the Hough lines are sorted and filtered into two sets, one for the left lane line and one for the right lane line. For each lane line, there are three filter criteria applied. First, the general slope of the lane is used. For the left lane, the slope is expected to be negative (because the 0,0 origin of the image is in the upper left), and the right lane is expecting a positive slope. Second, to filter out any noise from the hough detection (from line edges that appear horizontal, or other artifacts in the road which fall in the ROI), we restrict the slope-filter to the right or left half of the image. Finally, we don't expect extreme angles from the lane lines, so a third filter is set to only capture slopes that are within an acceptable range (between 0.45 and 0.90 - positive or negative respectively).

       2. Process lane lines for single frame
          
          At this point, we have the filtered Hough lines for the current frame, sorted into sets of points (all of the endpoints for the Hough lines) for the left lane and right lane.

          We now take the set of points, and calculate the slope and intercept using the cv2.fitline() function. This will fit a line to the set of points, and return the direction vector and a point on the line (this was initially confusing, but provides all the infromation we require). From the direction vector, we can calculate the slope yv/xv, and then solve for the intercept using y=mx+b, the calculated slope, and the point on the line that was provided by cv2.fitline().

          A key point to this implementation is that the current frame's lane lines are processed against a history of previous lane lines. This is useful, because we know that the lane lines will not drastically differ between frames. Exploiting this information helps to smooth out any jitter in the lane line detection and fill in information for garbage frames. 

          A FIFO of the previous N frame's lane line slope/intercept pairs are kept in a set of global variables. For this implementation, N=5. A deeper history will further smooth any jitter, but will also be slower to react to large changes in the lane line positioning.

          The average slope/intercept is a simple calculation that sums all slopes and all intecepts and divides by N. This is implemented in the slope_intercept() helper function.

          The algorithm for processing the current frame's lane line is:

             * If new lane line is within acceptable parameters (Slope is +/- 30% of the average slope of the saved N frames and the intercept is +/- 30 of the acerage intercept of the saved N frames), then the oldest slope/intercept is popped from the FIFO and the current frames slope/intercept is pushed to the back of the FIFO. The resulting lane line is computed as the average slope/intercept of the N frames in the FIFO (this now includes the current frame)

             * If the new lane is not within the above parameters, the resulting lane line is the average slope/intercept of the N frames in the FIFO (this DOES NOT include the current frame's slope/intercept)

             * If the FIFO has fewer than 10 items (we are still "training"), then the resulting lane line is the current frame's slope/intercept
          
          This processing is contained in the process_lane_line() helper function.

          It is possible that after the intial filter of the Hough lines, there are not enough points to fit a line, if this is the case, the average slope/intercept of the FIFO is returned. This happens if the lane line is particularly obscured or missing.

          Once we have the slope/intercept for the two lane lines, we compute the endpoints using y=mx+b, with y being the bounds of the ROI and draw the lines into a blank image. 


6. Overlay Hough lines 

   This stage is fairly simple, we just composite the resulting lane line image onto the original image using the proivided weighted_img() function.


Pipeline run on test images:

SolidWhiteCurve

![alt text][solidWhiteCurveFound]

SolidWhiteRight

![alt text][solidWhiteRightFound]

solidYellowCurve

![alt text][solidYellowCurveFound]

solidYellowCurve2

![alt text][solidYellowCurve2Found]

solidYellowLeft

![alt text][solidYellowLeftFound]

whiteCareLaneSwitch

![alt text][whiteCarLaneSwitchFound]

Video output can be found in test_videos_output/


### 2. Identify potential shortcomings with your current pipeline


I believe there are a few shortcomings with the current pipeline:

   1. Processing time
   
      The processing of a single frame needs to pass through the entire pipeline before the next frame can be processed. This causes the processing of the input data to occur slower than real-time (24fps or 30fps). The implication is that if deployed, likely there would be a significant amount of frames dropped.

      Given that this algorithm saves previous frames in a FIFO and assumes that there isn't a large deviation of the lane line between frames, large numbers of dropped frames could cause issues with this assumption.

   2. Susceptible to loss of lane

      There are two possible issues with losing the lane. First, we have made an assumption of where the lanes lines are likely to be and masked off data which is outside that region. If for some reason, the lane lines shift out of this region, we will no longer detect that lane line.

      Additionally, with this particular implementation, the lane line information for "garbage frames" is not stored in the history. If the lane line drops out for several frames, it may be difficult for the current implementation to re-track the lane line. This can be seen when the left lane in the challenge video breifly drops out, and is later reaquired, but several frames pass before the reacquisition occurs.

   3. Predicting lines not curves

      Using this method, we are detecting lane lines, however, real lanes often curve to the left or right. This method works well for highway scenarios where turns are very gradual, but I think it would suffer for windy roads that had more frequently and more severe turns.

   4. Reliance on painted lane lines

      Detecting the lane lines works well when the lane lines are correctly and clearly painted on the road. This is often not the case on secondary roads, which may not have lane lines painted, construction zones, which may have inaccurate or missing lane lines, or older roads where the painted lines may be obscured or partially missing. 

      I would suspect this also has some shortcomings when the lane forks or splits as it would for an exit/entrance ramp on the highway.

### 3. Suggest possible improvements to your pipeline

   1. Processing time

      Because this is implemented as a pipeline, I believe the pipe stages could be processed in parallel. This would incur an initial latency for the first lane lines to be 