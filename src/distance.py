import cv2
import numpy as np
import glob

CHECKERBOARD_SIZE = (6, 8)
SQUARE_SIZE = 0.25  # in meters

# 3D points of checkerboard in world frame
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)             # 6*8 3D points (z=0)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)   # meshgrid for x & y 
objp *= SQUARE_SIZE                                                                       # resolution 

# Arrays to store object points and image points from all checkerboard images
objpoints = []  # 3d points in world frame
imgpoints = []  # 2d points in image plane

calibration_images = glob.glob("../calibration/*.png")

for fname in calibration_images:
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
    
    # If found, add object points and image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
        cv2.imshow("Calibration", img)
        cv2.waitKey(500)

# Calibration
ret, intrinsic_mat, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Intrinsic matrix:")
print(intrinsic_mat)
# print("Image Size:", gray.shape[::-1])