# Project Details
These programs change the color of the image based on a histogram computed from a window in the image. The window is specified in terms of the normalized coordinates w1 h1 w2 h2, where the window upper left point is (w1,h1), and its lower right point is (w2,h2). For example, w1=0,h1=0,w2=1,h2=1 is the entire image, and w1=0.3,h1=0.3,w2=0.7,h2=0.7 is is window in the center of the image. The provided example program proj1b.py shows how to go over the pixels of this window

# First Program
Write a program that gets as input a color image, performs linear scaling in the Luv domain, and writes the scaled image as output. The scaling in Luv should stretch only the luminance values. You are asked to apply linear scaling that would map the smallest L value in the specified window and all values below it to 0, and the largest L value in the specified window and all values above it to 100.

# Second Program
Write a program that gets as input a color image, performs histogram equalization in the Luv domain, and writes the scaled image as output. Histogram equalization in Luv is applied to the luminance values, as computed in the specified window. It requires a discretization step, where the real-valued L is discretized into 101 values. As in the first program, all L values below the smallest L value in the window should be mapped to 0, and all L value above the largest L value in the window should be mapped to 100.
