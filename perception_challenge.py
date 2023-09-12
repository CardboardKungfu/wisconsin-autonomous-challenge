import numpy as np
import cv2 as cv

# Create resize function so that window isn't too large
# Changing the scale will affect the image. If the ratio is too large, the blurring isn't effective, if it's too small, some cones are lost
def resizeImg(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Read in image
img_raw = cv.imread("red_cones.png")
assert img_raw is not None, "file could not be read, check with os.path.exists()"

img = resizeImg(img_raw)
cv.imshow("Cones", img)

# Create blank to draw our mask on
blank = np.zeros(img.shape, dtype='uint8')

# Create a mask by changing the colors to create contrast, then use that to threshold based on a red-ish color value. 
img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_HSV = cv.cvtColor(img_RGB, cv.COLOR_RGB2HSV)

lower = np.array([0,135,160])
upper = np.array([179,255,255])
mask = cv.inRange(img_HSV,lower,upper)
img_masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow("Masked cones", img_masked)

# MORPH_OPEN uses erosion followed by dilation, specifically in that order. It removes the unwanted artifacts in the top middle created by the red lights, alarm clock, and the reflection
kernel = np.ones((5, 5))
img_opened = cv.morphologyEx(img_masked, cv.MORPH_OPEN, kernel)
# cv.imshow("Opened Image", img_opened)

median_blur = cv.medianBlur(img_opened, 7)
# cv.imshow("Median Blur", median_blur)

canny = cv.Canny(median_blur, 125, 175)
# cv.imshow("Canny Edges", canny)

# Draw our contours
contours, hierarhies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found:')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
# cv.imshow('Contours Drawn', blank)

# Contours on the left and right
vertical_center = blank.shape[1] / 2

cones_left = []
cones_right = []

# Use the moments library to find the center of our contours, and draw them
# While we're looping, might as well sort the contours into left and right
for i in contours:
	M = cv.moments(i)
	if M['m00'] != 0:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv.drawContours(blank, [i], -1, (0, 255, 0), 2)
		cv.circle(blank, (cx, cy), 2, (0, 0, 255), -1)
		# print(f"x: {cx} y: {cy}")
		if cx < vertical_center:
			cones_left.append(i[0])
		else:
			cones_right.append(i[0])

# cv.imshow("Contours with Centroid", blank)

left_fitline = cv.fitLine(np.array(cones_left), cv.DIST_L2, 0, 0.01, 0.01)
right_fitline = cv.fitLine(np.array(cones_right), cv.DIST_L2, 0, 0.01, 0.01)

# compute t0 for y=0 and t1 for y=img.shape[0]: (y-y0)/vy
t0 = (0 - left_fitline[3]) / left_fitline[1]
t1 = (blank.shape[0] - left_fitline[3]) / left_fitline[1]

# plug into the line formula to find the two endpoints, p0 and p1
# to plot, we need pixel locations so convert to int
p0 = (left_fitline[2:4] + (t0 * left_fitline[0:2]))
p1 = (left_fitline[2:4] + (t1 * left_fitline[0:2]))

# draw the line. For my version of opencv, it wants tuples so we
# flatten the arrays and convert
# args: cv2.line(image, p0, p1, color, thickness)
cv.line(blank, p0, p1, (0, 255, 0), 10)
cv.imshow("Drawn lines", blank)

cv.waitKey(0)
cv.destroyAllWindows()