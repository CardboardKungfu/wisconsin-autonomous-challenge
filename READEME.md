Wisconsin-Autonomous Perception Challenge

Methodology

What did you try and why do you think it did not work?
- This is my first project in OpenCV, so initially I first thought that I would need to train a model on recognizing cones, use it to label the cones, and then draw a line through their centers of mass. However, I did a lot of Googling and discovered that Hue Saturation Value (HSV) was a thing. I implemented HSV until I was content with my mask. However, I ran into the issue of too similar of colors fudging up my mask. After implementing a median blur, I found that it cleaned up the artifacts nicely. I did lose some data (the seventh cone on the left side) but I was content with the approximation.
- 

What libraries are used?
- OpenCV
- Numpy