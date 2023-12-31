from distance import calc_distance
from lane import detect_lane_markers
from detection import detect_bbox
from detection import preprocess
import cv2


INPUT_IMG_PATH = "../resource/test_img1.jpg"


# returns center point of the bottom bounding box edge
def get_bottom(bbox):
    cx, cy, h = bbox[0, 0], bbox[0, 1], bbox[0, 3] # i think there will be an error here
    return cx, cy + h/2   


if __name__=="__main__":

    # get img path
    input_image = cv2.imread(INPUT_IMG_PATH)
    shape = preprocess(input_image).shape
    print(shape)

    # call detect_bbox - bbox
    bbox = detect_bbox(input_image)
    print(bbox)
    # call get_bottom on bbox -> (u,v)
    u, v = get_bottom(bbox)
    print(u, v)

    u = u*960/shape[3]
    v = v*540/shape[2]
    print(u, v)
    # call distance on (u,v) -> (x, y)
    x_car, y_car = calc_distance(u, v)
