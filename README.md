# Wisconsin Autonomous Perception Challenge

## Methodology
1. This is my first project in OpenCV, so initially I first thought that I would need to train a model on recognizing cones, use it to label the cones, and then draw a line through their centers of mass. However, I did a lot of Googling and discovered that Hue Saturation Value (HSV) was a thing. 
2. First thing I did was create a rescale function. This served two purposes: first, it allowed me to see the image on my small laptop screen, and second, it worked as ad hoc erosion/dilation (admittedly, this was unintended, but welcome).
3. I used HSV to generate a mask. I used a color picker online to isolate the colors in the cones and do my best to distinguish them from the color in the alarm clock, the exit sign, and their reflections.
4. The exit sign and alarm clock were still persisting, so after a while of looking, I found cv2.morphologyEx, specifically I used the MORPH_OPEN flag that erodes and then dilates the image. After some bluring, it cleaned the artifacts right up.
5. I used Canny edge detection and then found the contours based on that.
6. I used cv2.fitLine() to create a line of best fit and then draw a line that extended to the borders based on the return values. Finally, I saved the image as a .png

## What did you try and why do you think it did not work?
- I wasn't sure of how to approach this. At first, I looked (barely) into training a model to recognize the cones and then get their positions from that. However, I discovered HSV and went down that route.
- My largest struggle was working out cv2.fitLine(). The documentation was just vague enough for me to have to experiment to get it working. At first I thought it returned a second point along the line, so when I tried to draw a line with it (after using int(), which set the values to zero), I kept getting a line drawn to the origin. This took me a while to figure out until I finally worked out that fitLine returns a point on the line as well as a normalized unit vector. With this, I was able to use some deceptively tricky math to workout the borders

## What libraries are used?
- OpenCV
- Numpy
