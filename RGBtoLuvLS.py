#created by Nishant Shekhar (nxs167130)

import cv2
import numpy as np
import sys

# if(len(sys.argv) != 7) :
#     print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
#     print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
#     print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
#     sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]
Llist=[]


if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)

if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()


rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

u_w = (4*0.95/(0.95+ 15 + 3*1.09))
v_w = (9/(0.95+ 15 + 3*1.09))


maxL = -1
minL = 101

def inverseGamma(t):
    if (t < 0.03928):
        return t/12.92
    else:
        return np.power(((t + 0.055)/1.055),2.4)


def gamma(d):
    if (d < 0.00304):
        return 12.92*d
    else:
        return 1.055*(np.power(d, 1/2.4))- 0.055

def gammaClip(g):
    if(g < 0):
        return 0
    elif(g > 1):
        return 1
    else:
        return g
    

#Linear RGB to XYZ conversion matrix
LRGB2XYZMatrix = [[0.412453, 0.357580, 0.180423],
                  [0.212671, 0.715160, 0.072169],
                  [0.019334, 0.119193, 0.950227]]

#XYZ to Linear RGB conversion matrix
XYZ2LRGBMatrix = [[3.240479, -1.53715, -0.498535],
                  [-0.969256, 1.875991, 0.041556],
                  [0.055648, -0.204043, 1.057311]]


LsRGBMatrix = np.zeros([rows, cols, bands], dtype=np.float16)

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]

        #sRGB to Non-Linear sRGB
        non_linear_B = b / 255.0
        non_linear_G = g / 255.0
        non_linear_R = r / 255.0

        #Non-Linear sRGB to Linear sRGB
        linear_B = inverseGamma(non_linear_B)
        linear_G = inverseGamma(non_linear_G)
        linear_R = inverseGamma(non_linear_R)


        LsRGBMatrix[i, j] = [linear_R, linear_G, linear_B]


XYZMatrix = np.zeros([rows, cols, bands], dtype=np.float16)
#Linear sRGB to XYZ
for i in range(0, rows) :
    for j in range(0, cols) :
        RGB = LsRGBMatrix[i, j]
        XYZMatrix[i,j] = np.dot(LRGB2XYZMatrix, RGB)


LuvMatrix = np.zeros([rows, cols, bands], dtype=np.float16)
for i in range(0, rows) :
    for j in range(0, cols) :
        X, Y, Z = XYZMatrix[i, j]
        # Calculate the L
        L = 116 * np.power(Y, 1/3.0) - 16.0 if Y > 0.008856 else 903.3 * Y

        d = X + 15*Y + 3*Z
        if(d <= 0):
            d = 0.1

        # Calculate the u 
        u_temp = (4*X)/d
        u = 13*L*(u_temp - u_w)
        
        # Calculate the v
        v_temp = (9*Y)/d
        v = 13*L*(v_temp - v_w)

        LuvMatrix[i,j] = [L, u, v]


#To find the max and min L in a given window
for i in range(H1, H2) :
    for j in range(W1, W2) :
        L, u, v = LuvMatrix[i, j]
        
        if(L >= maxL):
            maxL = L

        if(L <= minL):
            minL = L 

   
#Linear scaling in Luv Domain
for i in range(0, rows) :
    for j in range(0, cols) :
        L, u, v = LuvMatrix[i, j]
        Llist.append(int(L))     
        L = ((L - minL)*100)/(maxL - minL)
        LuvMatrix[i,j] = [L, u, v]    


Luv2XYZMatrix = np.zeros([rows, cols, bands], dtype=np.float16)

#Luv to XYZ conversion
for i in range(0, rows) :
    for j in range(0, cols) :
        L, u, v = LuvMatrix[i, j]

        if(L == 0):
            u_temp = 0
            v_temp = 0
        else:
            u_temp = (u + 13*u_w*L)/(13*L)
            v_temp = (v + 13*v_w*L)/(13*L)

        if(L > 7.9996):
            Y = np.power((L + 16)/116, 3)
        else:
            Y = L/903.3
        
        if(v_temp == 0):
            X = 0
            Z = 0
        else:
            X = Y*2.25*(u_temp/v_temp)
            Z = (Y*(3 - 0.75*u_temp- 5*v_temp))/v_temp
        
        Luv2XYZMatrix[i,j] = [X, Y, Z]

XYZ2LsRGBMatrix = np.zeros([rows, cols, bands], dtype=np.float16)
LsRGB2NLsRGBMatrix = np.zeros([rows, cols, bands], dtype=np.float16)

for i in range(0, rows) :
    for j in range(0, cols) :
        XYZ = Luv2XYZMatrix[i, j]
        #Linear RGB from XYZ
        XYZ2LsRGBMatrix[i, j] = np.dot(XYZ2LRGBMatrix, XYZ)

        R, G, B = XYZ2LsRGBMatrix[i, j]

        #Non Linear RGB from Linear RGB 
        non_linear_R = gammaClip(gamma(R))
        non_linear_G = gammaClip(gamma(G))
        non_linear_B = gammaClip(gamma(B))

        LsRGB2NLsRGBMatrix[i, j] = [non_linear_R, non_linear_G, non_linear_B]


scaledOutput = np.zeros([rows, cols, bands], dtype=np.uint8)

#final Scaled output in sRGB
for i in range(0, rows) :
    for j in range(0, cols) :
        
        n_linear_R, n_linear_G, n_linear_B = LsRGB2NLsRGBMatrix[i, j]

        r = np.rint(n_linear_R*255)
        g = np.rint(n_linear_G*255)
        b = np.rint(n_linear_B*255)
      
        scaledOutput[i,j] = [b, g, r]


cv2.imshow('Input Image', inputImage)
cv2.imshow('Scaled Output', scaledOutput)
cv2.imwrite(name_output, scaledOutput)    



# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
