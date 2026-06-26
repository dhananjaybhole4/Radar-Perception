import numpy as np

bbox1 = [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0]
bbox2 = [10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0]

def compute_3d_iou(bbox1: np.ndarray,
                    bbox2: np.ndarray):
    """Calculates intersection over union of two bounding boxes
    bbox format: [x, y, z, l, w, h, heading]
    x: x coordinate
    y: y coordinate
    z: z coordinate
    l: length of bbox
    w: width of bbox
    h: height of bbox
    heading: the direction of bbox

    Args:
        bbox1 (np.ndarray): _description_
        bbox2 (np.ndarray): _description_
    """
    # calculating extreme points of the 3D bbox
    high_bbox1 = np.array([bbox1[0] + bbox1[3]/2, bbox1[1] + bbox1[4]/2, bbox1[2] + bbox1[5]/2])
    low_bbox1 = np.array([bbox1[0] - bbox1[3]/2, bbox1[1] - bbox1[4]/2, bbox1[2] - bbox1[5]/2])

    high_bbox2 = np.array([bbox2[0] + bbox2[3]/2, bbox2[1] + bbox2[4]/2, bbox2[2] + bbox2[5]/2])
    low_bbox2 = np.array([bbox2[0] - bbox2[3]/2, bbox2[1] - bbox2[4]/2, bbox2[2] - bbox2[5]/2])

    x = np.minimum(high_bbox1, high_bbox2) - np.maximum(low_bbox1, low_bbox2)

    # if the intersection is negative that means there is no intersection
    intersection_dim = np.maximum(0, x)

    # vol of intersection
    volume_intersection = intersection_dim[0]* intersection_dim[1]*intersection_dim[2]

    # vol of union
    volume_union = bbox1[3]*bbox1[4]*bbox1[5] + bbox2[3]*bbox2[4]*bbox2[5] - volume_intersection

    iou = volume_intersection / volume_union

    return iou



if __name__ == "__main__":
    iou = compute_3d_iou(bbox1, bbox2)
    print(iou)