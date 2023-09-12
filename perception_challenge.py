import numpy as np
import cv2 as cv

def resizeImg(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img_raw = cv.imread("red_cones.png")
assert img_raw is not None, "file could not be read, check with os.path.exists()"

img = resizeImg(img_raw)
cv.imshow("Cones", img)

blank = np.zeros(img.shape, dtype='uint8')

img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_HSV = cv.cvtColor(img_RGB, cv.COLOR_RGB2HSV)

lower = np.array([0,135,160])
upper = np.array([179,255,255])
mask = cv.inRange(img_HSV,lower,upper)
img_masked = cv.bitwise_and(img, img, mask=mask)
# cv.imshow("Masked cones", img_masked)

kernel = np.ones((5, 5))
img_opened = cv.morphologyEx(img_masked, cv.MORPH_OPEN, kernel)
# cv.imshow("Opened Image", img_opened)

median_blur = cv.medianBlur(img_opened, 7)
# cv.imshow("Median Blur", median_blur)

canny = cv.Canny(median_blur, 125, 175)
# cv.imshow("Canny Edges", canny)

contours, hierarhies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found:')

cv.drawContours(blank, contours, -1, (0,0,255), 1)
# cv.imshow('Contours Drawn', blank)

for i in contours:
	M = cv.moments(i)
	if M['m00'] != 0:
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		cv.drawContours(blank, [i], -1, (0, 255, 0), 2)
		cv.circle(blank, (cx, cy), 2, (0, 0, 255), -1)
	
cv.imshow("Contours with Centroid", blank)

cv.waitKey(0)
cv.destroyAllWindows()