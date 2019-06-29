import numpy as np
import cv2
import  imutils
from PIL import Image

# Read the image file
image = cv2.imread('Car_Image_1.jpg')

# Resize the image - change width to 500
image = imutils.resize(image, width=500)

# Display the original image
#cv2.imshow("Original Image", image)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("1 - Grayscale Conversion", gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
#cv2.imshow("2 - Bilateral Filter", gray)

# Find Edges of the grayscale image
edged = cv2.Canny(gray, 170, 200)
#cv2.imshow("4 - Canny Edges", edged)

# Find contours based on Edges
result = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#result = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnts, new = result if len(result) == 2 else result[1:3]
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] #sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
NumberPlateCnt = None #we currently have no Number plate contour

# loop over our contours to find the best possible approximate contour of number plate
count = 0
for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            pl=NumberPlateCnt
            print(NumberPlateCnt)
            if(pl[0][0][1]+10>pl[2][0][1] or pl[0][0][0]+40>pl[2][0][0]):
                continue
            filter_img = image[pl[0][0][1]:pl[2][0][1],pl[0][0][0]:pl[2][0][0]]
            cv2.imshow("Number Plate Detected", filter_img)
            Number=pytesseract.image_to_string(filter_img,lang='eng')
            print("Number is :",Number)
            cv2.waitKey(0)
            cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)


cv2.imshow("Final Image With Number Plate Detected", image)

cv2.waitKey(0) #Wait for user input before closing the images displayed
