import cv2
import numpy as np
import glob

CALIBRATION_PATH = "../calibration/*.png"

def calc_distance(u, v):

    CHECKERBOARD_SIZE = (6, 8)
    SQUARE_SIZE = 0.25  # in meters

    # 3D points of checkerboard in world frame
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)             # 6*8 3D points (z=0)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)   # meshgrid for x & y 
    objp *= SQUARE_SIZE                                                                       # resolution 

    # Arrays to store object points and image points from all checkerboard images
    objpoints = []  # 3d points in world frame
    imgpoints = []  # 2d points in image plane

    calibration_images = glob.glob(CALIBRATION_PATH)

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
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
            # cv2.imshow("Calibration", img)
            # cv2.waitKey(500)

    # Calibration
    ret, intrinsic_mat, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print("Intrinsic matrix:")
    # print(intrinsic_mat)
    # print("Image Size:", gray.shape[::-1])

    # TODO - calc distance
    cam_coords = np.linalg.inv(intrinsic_mat) @ np.array([u, v, 1]).reshape(3,1)
    print("Cam coords: ", cam_coords)
    R = np.array([  [ 0, -1,  0],
                    [ 0,  0, -1],
                    [ 1,  0,  0]])
    h = get_camera_h()
    print("h: ", h, "m")
    T = np.array([0, h, 0]).reshape(3,1)
    l = h / cam_coords[1]
    car_coords = np.linalg.inv(R) @ (l*cam_coords - T)
    print("Car coords: ", car_coords)
    print("Lambda: ", l)
    x_car, y_car = car_coords[0], car_coords[1]
    print(x_car, y_car)
    
    return x_car, y_car


def get_camera_h():
    CHECKERBOARD_SIZE = (6, 8)
    SQUARE_SIZE = 0.25  # in meters

    # 3D points of checkerboard in world frame
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)             # 6*8 3D points (z=0)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)   # meshgrid for x & y 
    objp *= SQUARE_SIZE                                                                       # resolution 

    # Arrays to store object points and image points from all checkerboard images
    objpoints = []  # 3d points in world frame
    imgpoints = []  # 2d points in image plane

    calibration_images = glob.glob(CALIBRATION_PATH)

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
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD_SIZE, corners, ret)
            # cv2.imshow("Calibration", img)
            # cv2.waitKey(500)

    # Calibration
    ret, intrinsic_mat, distortion_coeff, rotation_vec, translation_vec = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Intrinsic matrix:")
    print(intrinsic_mat)

    u, v = 664, 493 # Need to get from the image after clicking 

    cam_coords = np.linalg.inv(intrinsic_mat) @ np.array([u, v, 1]).reshape(3,1)
    R = np.array([  [ 0, -1,  0],
                [ 0,  0, -1],
                [ 1,  0,  0]])
    car_coords = np.array([0.4, 0 , 0]).reshape(3,1) # Ground truth d = 40cm

    save = R @ car_coords
    T = cam_coords - (R @ car_coords)/save[-1]
    
    T = T*save[-1]
    h = T[1][0]
    return h


def click_event(event, x, y, flags, params):
  
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
  
    # checking for right mouse clicks     
    if event==cv2.EVENT_RBUTTONDOWN:
  
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)



if __name__=='__main__':

    # To get distance if u and v is knownn
    # '''
    u, v = 674, 414
    calc_distance(u,v)
    # '''
    
    # To find height of camera mount
    '''
    h = get_camera_h()
    print("height: ", h)
    '''

    # To get camera coordinates [u, v] from image
    '''
    img = cv2.imread('../resource/test_car_x60cm.png', 1)
  
    # displaying the image
    cv2.imshow('image', img)
  
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)
  
    # close the window
    cv2.destroyAllWindows()
    '''

    
