from distance import calc_distance
from lane import detect_lane_markers
from detection import detect_bbox
import cv2


INPUT_IMG_PATH = "../imgs/input_img.png"


# returns center point of the bottom bounding box edge
def get_bottom(bbox):
    cx, cy, _, h = bbox
    return cx, cy + h/2   


if __name__=="__main__":

    # get img path
    input_image = cv2.imread(INPUT_IMG_PATH)

    # call detect_bbox - bbox
    bbox = detect_bbox(input_image)

    # call get_bottom on bbox -> (u,v)
    u, v = get_bottom(bbox)

    # call distance on (u,v) -> (x, y)
    x_car, y_car = calc_distance(u, v)