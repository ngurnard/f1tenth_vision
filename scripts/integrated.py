from distance import calc_distance
from lane import detect_lane_markers
from detection import detect_bbox


# returns center point of the bottom bounding box edge
def get_bottom(bbox):
    cx, cy, _, h = bbox
    return cx, cy + h/2   


if __name__=="__main__":

    # get img path
    input_image = None
    img_path = None

    # call lane
    lane_img = detect_lane_markers(input_image)
    
    # call detect_bbox - bbox
    bbox = detect_bbox(img_path)

    # call get_bottom on bbox -> (u,v)
    v, u = get_bottom(bbox)

    # call distance on (u,v) -> (x, y)
    x_car, y_car = calc_distance(u, v)
