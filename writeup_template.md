# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The lane finding pipeline is comprised of 5 stages:

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


![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
