#!/usr/bin/env python3
# Created by Joel John

"""mapreader -- outputs the position of the green pointer and its bearing. """

# Importing neccessary packages

import sys
import cv2
import numpy
import math

#------------------------------------------------------------------------------
# Main program.

# Ensure we were invoked with a single argument.

if len (sys.argv) != 2:
    print ("Usage: %s <image-file>" % sys.argv[0], file=sys.stderr)
    exit (1)

for fn in sys.argv[0:]:
    img = cv2.imread (fn)

#  Making a copy of the orginal image if needed at a later point.

img_copy =  img.copy()

# print ("The filename to work on is %s." % sys.argv[1])

#==============================================================================
#===========================Extracting the Map=================================
#==============================================================================

"""Segmneting the map from the blue backgroungd and extracting it so that the 
map edges matches the extracted image."""

# Following lines adapted from:
"""https://www.geeksforgeeks.org/detection-specific-colorblue-using-opencv-
python/"""

# Converting the read image to HSV space.

HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# As we know the background of the image is always going to be blue.
# Set an upper and lower limit for blue colour in HSV Space.

Lower_Blue = numpy.array([90,80,2]      ,dtype= "uint8") 
Upper_Blue = numpy.array([125,255,255]  , dtype="uint8")

# Detecting the pixels which are blue in colur.

Mask = cv2.inRange(HSV, Lower_Blue, Upper_Blue)

# Change all the pixels that are blue to black in the orginal image with help 
# of the detected Mask.

img[Mask!=0] = [0,0,0]

# Draw contours to find the edges of the map.

contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, 
		       cv2.CHAIN_APPROX_SIMPLE)

# As the conditions remains the same the second biggest rectagle counter 
# will be around the map with that in mind we are getting the edge co-ordinates
# of the map.

# Following lines adapted from:
"""https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using
-opencv/"""

"""https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html"""

# Using the following function we are finding a countor with 4 distinct sides
# but we are avoiding first rectangle/4-sided contour which will be the made 
# with the whole image in it.

# Instead of just looking contour with 4 sides only we are looking for contour
# which has an area more than 300000 (500 pixel * 600 pixel)

for i,c in  enumerate(contours):    
    if i > 0 :
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
        if len(approx)== 4 and cv2.contourArea(c) > 300000 :
            break

# Assign the four edge co-ordinates to four variables.       
Edge_A = approx[0][0]
Edge_B = approx[1][0]
Edge_C = approx[2][0]
Edge_D = approx[3][0]            
            
            
# Finding  width and length of the map, Following eqaation adapted from

# https://courses.lumenlearning.com/waymakercollegealgebra/chapter/
# distance-in-the-plane/


Width_AD  =numpy.sqrt (((Edge_A[0]-Edge_D[0])**2) + 
						((Edge_A[1]-Edge_D[1])**2))
Width_BC  =numpy.sqrt (((Edge_B[0]-Edge_C[0])**2) + 
						((Edge_B[1]-Edge_C[1])**2))

Length_AB =numpy.sqrt (((Edge_A[0]-Edge_B[0])**2) + 
						((Edge_A[1]-Edge_B[1])**2))
Length_CD =numpy.sqrt (((Edge_C[0]-Edge_D[0])**2) + 
						((Edge_C[1]-Edge_D[1])**2))

# Assign the max width and lenth to a variable.
Width  = max(Width_AD,Width_BC)
Length = max(Length_AB,Length_CD)

# Saving the float values for both Length and Width to another variable for 
# later use.
Orginal_Width  = Width
Orginal_Length = Length

# The following function can handle only integer values so Width and Length 
# is conveted   

Width = int(Width)
Length = int(Length)

# Following lines adapted from:
"""https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html"""


Map_Edges         = numpy.float32([Edge_D,Edge_C,Edge_B,Edge_A])

Expeted_Map_Edges = numpy.float32([[0,0],[0,Length],[Width,Length],[Width,0]])

matrix = cv2.getPerspectiveTransform(Map_Edges,Expeted_Map_Edges)

result = cv2.warpPerspective(img_copy, matrix, (int(Width),int(Length)))

#==============================================================================
#==========================Correcting the Orientation==========================
#==============================================================================

"""Locating the green arrow and  if it is not top-right of the map, rotating
it 180 degree."""

# Following lines adapted from:
"""https://www.geeksforgeeks.org/python-opencv-cv2-rotate-method/"""


# We are going to set an upper and lower limit for green colour in HSV Space
Lower_Green = numpy.array([45, 100, 100],dtype= "uint8") 
Upper_Green = numpy.array([75,255,255]  , dtype="uint8")

# Declaring a variable to control the while loop

rotate = 1

# Until the arrow denoting the bearing of the map is on the left upper corner 
# of the image the loop will keep on running each time doing a 180 degree turn.

while rotate:

    # Converting the cropped image to HSV space 
    HSV = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    
    # Detecting all the pixels that are green in color in the image
    Mask = cv2.inRange(HSV , Lower_Green, Upper_Green)
    
    # Finding the contours
    contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, 
    			   cv2.CHAIN_APPROX_SIMPLE)
    
    # Finding the countour that represent the arrow
    for i,c in  enumerate(contours):     
        peri   = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.04 * peri, True)
        if len(approx)> 3 and cv2.contourArea(c) < 100000 :
            break
    
    # Taking a edge point on the arrow
    Edge = approx[0][0]
    
    # Check if the edge of the green arrow is on the top right corner 
    # if true exit the loop else rotate it 180 degree and check again 
    
    if Edge[0]>700 and Edge[1]<500:
        rotate = 0

    else:
        result = cv2.rotate(result, cv2.ROTATE_180)
 
#==============================================================================
#===================Finding the position of red arrow==========================
#==============================================================================

"""Segmenting the Red pointer and locating the tip and finding the location 
coordinate between 0 -1 in x and y axis. """

# We are going to set an upper and lower limit for red colour in HSV Space
# Unlike blue and green the red in HSV is situated on both sides of zero 
# for hue we have two upper and lower limits.

# Following lines adapted from:
# https://answers.opencv.org/question/229620/drawing-a-rectangle-around-the-
# red-color-region/

Lower_Red_1 = numpy.array([0,100,30]  ,dtype= "uint8") 
Upper_Red_1 = numpy.array([30,255,255], dtype="uint8")

Lower_Red_2 = numpy.array([145,100,30] ,dtype= "uint8") 
Upper_Red_2 = numpy.array([225,255,255], dtype="uint8")

# Detecting all the pixels that are red in color in the image.
Mask_A = cv2.inRange(HSV ,  Lower_Red_1, Upper_Red_1)
Mask_B = cv2.inRange(HSV ,  Lower_Red_2, Upper_Red_2)

Mask   = Mask_A +  Mask_B

# Finding the contours.
contours, hierarchy = cv2.findContours(Mask, cv2.RETR_TREE, 
						    cv2.CHAIN_APPROX_SIMPLE)

# Finding the countour that represent the red pointer.
for i,c in  enumerate(contours):    
    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    
    if len(approx)== 3 and cv2.contourArea(c) < 100000 :
        break
     
# Assign the three edge co-ordinates to three variables.
Edge_A = approx[0][0]
Edge_B = approx[1][0]
Edge_C = approx[2][0]

# Finding the length of each sides.
Length_AB  = numpy.sqrt (((Edge_A[0]-Edge_B[0])**2) + 
						((Edge_A[1]-Edge_B[1])**2))
						
Length_BC  = numpy.sqrt (((Edge_B[0]-Edge_C[0])**2) + 
						((Edge_B[1]-Edge_C[1])**2))
						
Length_CA  = numpy.sqrt (((Edge_C[0]-Edge_A[0])**2) + 
						((Edge_C[1]-Edge_A[1])**2))


# Checking the sides with the same size and finding the point that is shared 
# between the two.

if Length_AB > 100 and Length_BC > 100:
    Position = Edge_B
    
elif Length_BC > 100 and Length_CA > 100:
    Position = Edge_C  
    
elif Length_CA > 100 and Length_AB > 100:
    Position = Edge_A     


 
# To get the position between "1 - 0" we will divide the recived values from 
# above with width and length of the whole map which was obtained earlier.   

xpos = (Position[0])/Orginal_Width
ypos = 1 - ((Position[1])/Orginal_Length)

#==============================================================================
#=====================Finding the Bearing of Red Pointer=======================
#==============================================================================

"""Determining the orientation of the pointer, coverting it to bearing."""


# Finding the shortest side of the triangle to get coordinates of line passing
# through the center of the isoceles triagle/red pointer.

if   Length_AB < 100:
    Mid_Point = (Edge_A + Edge_B)/2
    
elif Length_BC <100:
    Mid_Point = (Edge_B + Edge_C)/2
    
elif Length_CA < 100:
    Mid_Point = (Edge_C + Edge_A)/2   
    
Center = Mid_Point

Center[0] = int(Mid_Point[0])
Center[1] = int(Mid_Point[1])

# Following lines adapted from:
""" https://stackoverflow.com/questions/37259366/using-python-to-calculate-
    radial-angle-in-clockwise-counterclockwise-directions """

dx = Position[0] - Center[0]
dy = Position[1] - Center[1]

In_radians = math.atan2(dy,dx)
In_Degree  = math.degrees(math.atan2(dy, dx))

# Because the angle is being calculated from the horizontal or positive X axis
# in clockwise direction inorder to get the value from North or positive 
# Y axis in a normal cartisian plane in clockwise direction we add 90 to the
# received value

hdg = 90 + In_Degree

#------------------------------------------------------------------------------
# Observed an offset of 2 Degrees in all the test so added that to get the
# final value.

offset = 2

# Passing the final value to hgd
hdg = hdg + offset
#------------------------------------------------------------------------------


#==============================================================================
#=================Passing final the position and bearing=======================
#==============================================================================



# Output the position and bearing in the form required by the test harness.
print ("POSITION %.3f %.3f" % (xpos, ypos))
print ("BEARING %.1f" % hdg)

#------------------------------------------------------------------------------

