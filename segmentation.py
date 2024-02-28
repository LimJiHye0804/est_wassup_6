import cv2
import numpy as np

# Load the image
img = cv2.imread("0a88a23fd9e2225216ed0ab0e2c1d01801be2e23b1f916fcab139fccd47811d2_남_20_기쁨_교통&이동수단(엘리베이터 포함)_20210125160548-002-004.jpg")

# Convert to grayscale and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(contours, key=cv2.contourArea)[-1]

# Create a mask and draw the contour in it
mask = np.zeros(img.shape[:2], dtype="uint8")
cv2.drawContours(mask, [cnt], -1, 255, -1)  # Draw filled contour

# Invert the mask so the contour area is 0 and the rest is 255

#mask_inv = cv2.bitwise_not(mask)

# Create an RGBA version of the original image
img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# Apply the inverted mask to the alpha channel
img_rgba[:, :, 3] = mask  # This makes the contour area opaque and the rest transparent
cv2.imwrite('output_with_transparency.png', img_rgba)
# Display the result
cv2.imshow("Masked Final with Transparency", img_rgba)

cv2.waitKey(0)
cv2.destroyAllWindows()