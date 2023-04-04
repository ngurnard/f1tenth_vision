import cv2

# cap = cv2.VideoCapture(cv2.CAP_V4L2)
cap = cv2.VideoCapture(4)    #6 - RGB, 4 - Depth

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 60)
i=0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    # cv2.imwrite('./captures/img'+str(i)+'.png', frame)
    # i+=1 

    # wait for key press
    if cv2.waitKey(500) == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()