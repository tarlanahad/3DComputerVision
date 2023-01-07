from sklearn.neighbors import KDTree
import numpy as np
from timeit import default_timer as timer

f = 3740  # in pixels focal length
B = 160 / 1000  # in milli meters baseline
scale = 3
dmin = 200


def project_disparity_to_3d(disparity, return_time=False):
    start = timer()

    points = []

    height, width = disparity.shape[:2]

    c = 0
    for y in range(height):  # 0 - height is the y axis index
        for x in range(width):  # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if (disparity[y, x] > 0):
                Z = (f * B) / (disparity[y, x] + dmin)

                X = ((x - width / 2) * Z) / f
                Y = ((y - height / 2) * Z) / f

                if c % 50 == 0:
                    points.append([X, Y, Z])

                c += 1

    if return_time:
        points, timer() - start
    return points


def project_disparity_to_3d_norm(disparity, k=3, return_time=False):
    start = timer()

    points = []

    naive_3d_points = np.array(project_disparity_to_3d(disparity))

    tree = KDTree(naive_3d_points, leaf_size=2)

    for p in naive_3d_points:
        dist, ind = tree.query(p.reshape(1, -1), k=k)
        p1 = naive_3d_points[ind[0][1]]
        p2 = naive_3d_points[ind[0][2]]
        points.append(np.cross(p1 - p, p2 - p))

    if return_time:
        points, timer() - start
    return points
