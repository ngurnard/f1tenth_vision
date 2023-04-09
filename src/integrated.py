from distance import calc_distance
from lane import detect_lane_markers
from detection import detect_bbox



def get_bottom(bbox):
    cx, cy, w, h = bbox
    return cx, cy + h/2   


if __name__=="__main__":
    pass
    # get img path
    # call lane
    # call detect_bbox - bbox
    # call get_bottom on bbox -> (u,v)
    # call distance on (u,v) -> (x, y)