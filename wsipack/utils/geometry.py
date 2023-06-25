import numpy as np


def merge_overlapping_bboxes(bboxes):
    candidate_count = 0
    while candidate_count < len(bboxes):
        candidate_count += 1
        overlap = False
        candidate_box = bboxes.pop(0)
        for index, compare_box in enumerate(bboxes):
            overlapping, new_bbox = merge_if_overlapping(candidate_box, compare_box)
            if overlapping:
                overlap = True
                candidate_count = 0
                bboxes.pop(index)
                bboxes.append(new_bbox)
                break
        if not overlap:
            bboxes.append(candidate_box)


def merge_if_overlapping(a, b):
    bottom = np.max([a[0],b[0]])
    top = np.min([a[0] + a[2], b[0] + b[2]])
    left = np.max([a[1],b[1]])
    right = np.min([a[1] + a[3], b[1] + b[3]])

    do_intersect = bottom < top and left < right
    #with shapely (same as result as with this custom code)
    # from shapely.geometry import Polygon
    # p1 = Polygon([(a[0], a[1]), (a[0], a[1]+a[3]), (a[0]+a[2], a[1]+a[3]), (a[0]+a[2], a[1])])
    # p2 = Polygon([(b[0], b[1]), (b[0], b[1]+b[3]), (b[0]+b[2], b[1]+b[3]), (b[0]+b[2], b[1])])
    # do_intersect2 = p1.intersects(p2)
    # if do_intersect != do_intersect2:
    #     print('!')
    #     raise

    if do_intersect:
        x_min = np.min([a[1],b[1]])
        y_min = np.min([a[0],b[0]])
        x_max = np.max([a[1]+a[3],b[1]+b[3]])
        y_max = np.max([a[0]+a[2],b[0]+b[2]])
        new_bbox = (y_min, x_min, y_max - y_min, x_max - x_min)
        return True, new_bbox

    return False, []