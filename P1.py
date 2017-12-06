
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[24]:


import math

### Globals to track state of previous images in video
prev_left_lane = []
prev_right_lane = []

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def slope_intercept(points):
    """
    Calculates the slope and intercept for a line defined by the provided endpoints
    """
    # This was a bit confusing at first, but fitline will return the direction vector, and a point
    # on the fitted line, this allows us to calculate the fitted slope and intercept
    [xv, yv, x, y] = cv2.fitLine(np.array(points, dtype=np.int32),cv2.DIST_L2,0,0.1,0.1)
    slope = yv/xv # rise/run - direction vector
    intercept = y - (slope * x) # solving for y = mx+b using returned point and calculate slope
    return (slope,intercept)

def average_lane_hist(prev_lane):
    """
    Calculates the average slope and intercept of the slop/intercept pairs in the provided list
    """
    n = 0
    slope = 0
    intercept = 0
    
    for [next_slope,next_intercept] in prev_lane:
        n = n + 1
        slope = slope+next_slope
        intercept = intercept+next_intercept
        
    return slope/n,intercept/n

def process_new_lane(prev_lane, lane_slope, lane_intercept):
    """
    Given a lane line in a new image, filter this for reasonble amount of change from lane line average
    (slope/intercept) and if the new lane line falls within reasonable limits, add to the running list of 
    saved previous lane lines. 
    
    The resulting lane line is:
    1) The new average slope and intercept of the saved frames, including the new frame
    2) If the new frame is not within acceptable deviation for the average, return the average
    3) If there are not enough saved frames, return the current lane line (still training)
    """
    ret_slope = 0
    ret_int = 0
    updated = 0
    
    if (len(prev_lane) >= 5):
#        print("using history ", len(prev_lane))
        avg_slope,avg_int = average_lane_hist(prev_lane)
        if (abs(lane_slope) < abs(avg_slope + avg_slope*.3) and
            abs(lane_slope) > abs(avg_slope - avg_slope*.3) and
            lane_intercept < avg_int + 30 and
            lane_intercept > avg_int - 30):
            updated = 1
            prev_lane.pop(0)
            prev_lane.append([lane_slope,lane_intercept])
        ret_slope,ret_int = average_lane_hist(prev_lane)
    else:
#        print("no history")
        ret_slope = lane_slope
        ret_int = lane_intercept
        prev_lane.append([lane_slope,lane_intercept])
        
    return ret_slope,ret_int,updated
        
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    # get some information from the image
    imshape = img.shape
    img_sizey = imshape[0]
    img_sizex = imshape[1]

    left_color = color
    right_color = color
    
    # separate line segments for left lane and right lane
    left_lane = []
    right_lane = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            line_slope = (y2-y1)/(x2-x1)
            # Initially filter found hough lines for any lines that have a slope/intercept that 
            # doesn't make sense to include in lane line detection
            #
            # Also, seperate line segments into left lane and right lane segments
            if ( line_slope < -0.45 and line_slope > -0.90 and x2 < img_sizex/2 and x1 < img_sizex/2):
                left_lane.append((x1,y1))
                left_lane.append((x2,y2))
            elif ( line_slope > 0.45 and line_slope < 0.90 and x2 > img_sizex/2 and x1 > img_sizex/2):
                right_lane.append((x1,y1))
                right_lane.append((x2,y2))

 # Debug - will draw all resulting points as solid circles into the image
 #   for lpoint in left_lane:
 #       cv2.circle(img, lpoint, 5, [255,0,0], 5)

 #   for rpoint in right_lane:
 #       cv2.circle(img, rpoint, 5, [0,0,255], 5)

    # check to see if our initial slope filter left us with enough points
    # if not, the we will just use the previous values
    if (len(left_lane) > 2):
        ll_slope,ll_intercept = slope_intercept(left_lane)
        ll_slope,ll_intercept,ll_updated = process_new_lane(prev_left_lane,ll_slope,ll_intercept)
        # Debug - will color the line purple if the current frame had unusable data, and we
        #         reverted to using the previous average
        # if (ll_updated != 1):
        #    left_color = [255,0,255]
    else: 
        ll_slope,ll_intercept = average_lane_hist(prev_left_lane)
        # Debug - will color the line yellow if resulting frame didn't provide enough points for a line
        # left_color = [255,255,0]
        
    # check to see if our initial slope filter left us with enough points
    # if not, the we will just use the previous values
    if (len(right_lane) > 2):
        rl_slope, rl_intercept = slope_intercept(right_lane)
        rl_slope, rl_intercept,rl_updated = process_new_lane(prev_right_lane,rl_slope,rl_intercept)
        # Debug - will color the line purple if the current frame had unusable data, and we
        #         reverted to using the previous average
        # if (rl_updated != 1):
        #    right_color = [255,0,255]
    else:
        rl_slope,rl_intercept = average_lane_hist(prev_right_lane)
        # Debug - will color the line yellow if resulting frame didn't provide enough points for a line
        # right_color = [255,255,0]

    # calculate the endpoints of the lines to draw
    ll_y1 = int(img.shape[0]*0.6)
    ll_y2 = img.shape[0]
    ll_x1 = int((ll_y1 - ll_intercept)/ll_slope)
    ll_x2 = int((ll_y2 - ll_intercept)/ll_slope)
    
#    ll_x1 = 0
#    ll_x2 = img.shape[1]
#    ll_y1 = ll_slope*ll_x1 + ll_intercept
#    ll_y2 = ll_slope*ll_x2 + ll_intercept

    rl_y1 = int(img.shape[0]*0.6)
    rl_y2 = img.shape[0]
    rl_x1 = int((rl_y1 - rl_intercept)/rl_slope)
    rl_x2 = int((rl_y2 - rl_intercept)/rl_slope)

#    rl_x1 = 0
#    rl_x2 = img.shape[1]
#    rl_y1 = rl_slope*rl_x1 + rl_intercept
#    rl_y2 = rl_slope*rl_x2 + rl_intercept

    cv2.line(img, (ll_x1,ll_y1), (ll_x2,ll_y2), left_color, thickness)
    cv2.line(img, (rl_x1,rl_y1), (rl_x2,rl_y2), right_color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[5]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[20]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

# Create a function to implement the lane finding pipeline
# img - input image
# returns output image with lane lines detected
def find_lane(img):
    # Capture some information about the image
    imshape = img.shape
    img_sizey = imshape[0]
    img_sizex = imshape[1]
    
    # Create some parameters to be used in the pipeline. Keeping these in one place to make them easier to tune.
    p2_gaussian_kernel_size = 5
 
    p3_canny_low_threshold = 50
    p3_canny_high_threshold = 200

    p4_x_left_bound = 450
    p4_x_right_bound = 550
    p4_y_bound = 325

    p5_rho = 1;
    p5_theta = np.pi/270
    p5_threshold = 3
    p5_min_line_len = 8
    p5_max_line_gap = 3
    
    # 1. Convert image to grayscale
    gray_img = grayscale(img)
    
    # 2. Apply guassian filter to smooth and remove noise
    gauss_img = gaussian_blur(gray_img, p2_gaussian_kernel_size)
    
    # 3. Perform Canny edge detection on image
    canny_img = canny(gauss_img, p3_canny_low_threshold, p3_canny_high_threshold)
    
    # 4. Mask a region of interest
    #   a. Create vertices which mark boundaries of a polygon
    vertices = np.array([[(0,img_sizey),(p4_x_left_bound,p4_y_bound),(p4_x_right_bound,p4_y_bound),(img_sizex,img_sizey)]], dtype=np.int32)
    #   b. Apply mask to canny image
    roi_canny_img = region_of_interest(canny_img,vertices)
    
    # 5. Apply Hough Transform
    hough_img = hough_lines(roi_canny_img, p5_rho, p5_theta, p5_threshold, p5_min_line_len, p5_max_line_gap)
    
    # 6. Overlay Hough lines
#    result = weighted_img(region_of_interest(hough_img,vertices), img)
    result = weighted_img(hough_img, img)
    
    return result
    
test_images = os.listdir("test_images/")
for test_image_file in test_images:
    prev_left_lane = []
    prev_right_lane = []
    print("Processing image test_images/", test_image_file)
    test_image = mpimg.imread('test_images/{0}'.format(test_image_file))
    result = find_lane(test_image)
    mpimg.imsave('test_images_output/{0}.png'.format(os.path.splitext(test_image_file)[0]),result)
    
    


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[8]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[21]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
   # Capture some information about the image
    imshape = image.shape
    img_sizey = imshape[0]
    img_sizex = imshape[1]
    
    # Create some parameters to be used in the pipeline. Keeping these in one place to make them easier to tune.
    p2_gaussian_kernel_size = 5
 
    p3_canny_low_threshold = 40
    p3_canny_high_threshold = 150

    p4_x_tlb = int(img_sizex/2 - (img_sizex/2)*0.1)
    p4_x_trb = int(img_sizex/2 + (img_sizex/2)*0.1)
    p4_x_blb = int(0+img_sizex*.1)
    p4_x_brb = int(img_sizex - img_sizex*.1)
    p4_y_tb = int(img_sizey*.6)
    p4_y_bb = int(img_sizey - img_sizey*.1)

    p5_rho = 1;
    p5_theta = np.pi/270
    p5_threshold = 3
    p5_min_line_len = 8
    p5_max_line_gap = 3
    
    # 1. Convert image to grayscale
    gray_img = grayscale(image)
    
    # 2. Apply guassian filter to smooth and remove noise
    gauss_img = gaussian_blur(gray_img, p2_gaussian_kernel_size)
    
    # 3. Perform Canny edge detection on image
    canny_img = canny(gauss_img, p3_canny_low_threshold, p3_canny_high_threshold)
    
    # 4. Mask a region of interest
    #   a. Create vertices which mark boundaries of a polygon
    vertices = np.array([[(p4_x_blb,p4_y_bb),(p4_x_tlb,p4_y_tb),(p4_x_trb,p4_y_tb),(p4_x_brb,p4_y_bb)]], dtype=np.int32)
    #   b. Apply mask to canny image
    roi_canny_img = region_of_interest(canny_img,vertices)
    
    # 5. Apply Hough Transform
    hough_img = hough_lines(roi_canny_img, p5_rho, p5_theta, p5_threshold, p5_min_line_len, p5_max_line_gap)
    
 # Debug - this will create a polygon that represents the ROI   
 #   roi_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)    
 #   cv2.line(roi_img, (p4_x_blb,p4_y_bb), (p4_x_tlb,p4_y_tb), [255,255,0], 2)
 #   cv2.line(roi_img, (p4_x_tlb,p4_y_tb), (p4_x_trb,p4_y_tb), [255,255,0], 2)
 #   cv2.line(roi_img, (p4_x_trb,p4_y_tb), (p4_x_brb,p4_y_bb), [255,255,0], 2)
 #   cv2.line(roi_img, (p4_x_blb,p4_y_bb), (p4_x_brb,p4_y_bb), [255,255,0], 2)
    
    #vertices = np.array([[(0,img_sizey),(p4_x_tlb,p4_y_tb),(p4_x_trb,p4_y_tb),(img_sizex,img_sizey)]], dtype=np.int32)
    # 6. Overlay Hough lines
#    result = weighted_img(region_of_interest(hough_img,vertices), image)
    result = weighted_img(hough_img, image)
 # Debug - will overlay the ROI polygon on the image
 #   result = weighted_img(roi_img, result)
    
    return result


# Let's try the one with the solid white lane on the right first ...

# In[25]:


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
prev_left_lane = []
prev_right_lane = []
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[26]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[27]:


yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
prev_left_lane = []
prev_right_lane = []
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[28]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[29]:


challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
prev_left_lane = []
prev_right_lane = []
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[15]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

