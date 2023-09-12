import numpy as np
import cv2 as cv

# Create resize function so that window isn't too large
# Changing the scale will affect the image. If the ratio is too large, the blurring isn't effective, if it's too small, some cones are lost
def resizeImg(frame, scale=0.25):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Use existing point to draw a line that extends to the borders of the window
def drawLine(img,x1,x2,y1,y2):
	m = (y2-y1) / (x2 - x1)
	w = img.shape[:2][1]
	px = 0
	py = -(x1-0) * m + y1

	qx = w
	qy = -(x2-w) * m + y2

	return cv.line(img, [int(px), int(py)], [int(qx), int(qy)], (0, 0, 255), 2)

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
kernel = np.ones((3, 3))
img_opened = cv.morphologyEx(img_masked, cv.MORPH_OPEN, kernel)
# cv.imshow("Opened Image", img_opened)

median_blur = cv.medianBlur(img_opened, 3)
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

# Arrays for the contours on the left and right of our image
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
			cones_left.append((cx, cy))
		else:
			cones_right.append((cx, cy))

# Fit line takes our points, finds a line of best fit, and then returns a 
# point on that line as well as a normalized vector
left_vx, left_vy, left_cx, left_cy = cv.fitLine(np.array(cones_left), cv.DIST_L2, 0, 0.01, 0.01)

left_x1 = int(left_cx[0])
left_x2 = left_cx + (left_vx * 10)
left_y1 = int(left_cy[0])
left_y2 = left_cy + (left_vy * 10)

drawLine(img, left_x1, left_x2, left_y1,left_y2)

right_vx, right_vy, right_cx, right_cy = cv.fitLine(np.array(cones_right), cv.DIST_L2, 0, 0.01, 0.01)

# print(f'right_vx: {right_vx}, right_vy: {right_vy}')

right_x1 = int(right_cx[0])
right_x2 = right_cx + (right_vx * 10)
right_y1 = int(right_cy[0])
right_y2 = right_cy + (right_vy * 10)

drawLine(img, right_x1, right_x2, right_y1, right_y2)

cv.imshow("Drawn lines", img)
cv.imwrite("answer.png", img)

cv.waitKey(0)
cv.destroyAllWindows()