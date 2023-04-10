import cv2
import numpy as np

def detect_lane_markers(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the yellow color range in HSV
    lower_yellow = np.array([20, 70, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the image to get only yellow pixels
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply a Gaussian blur to the mask
    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of contours:", len(contours))

    # Draw green edges around or green overlaps on the markers you detect
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)


    return image


if __name__=='__main__':
    input_image = cv2.imread('../resource/lane.png')
    output_image = detect_lane_markers(input_image)

    # display image
    cv2.imshow("Lane_Detection", output_image)
    cv2.waitKey(3000)

    cv2.imwrite('../imgs/lane_final.png', output_image)
  
