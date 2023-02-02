import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm


# https://vision.middlebury.edu/stereo/data/scenes2005/#description


scale = 10.0; 
focal_length = 3740;
camera_baseline = 160;
dmin = 200; 

def project_disparity_to_3d(disparity):

    points = []

    height, width = disparity.shape[:2]

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity
            if (disparity[y,x] > 0):
                
                Z = (focal_length * camera_baseline) / (disparity[y,x]*scale + dmin)

                X = ((x - width/2) * Z) / focal_length
                Y = ((y - height/2) * Z) / focal_length

                points.append([X,Y,Z])

    return points


def project_disparity_to_3d_norm(disparity, k=3):
    points = []

    naive_3d_points = np.array(project_disparity_to_3d(disparity))

    tree = KDTree(naive_3d_points, leaf_size=10)

    for p in naive_3d_points:
        dist, ind = tree.query(p.reshape(1, -1), k=k)
        p1 = naive_3d_points[ind[0][1]]
        p2 = naive_3d_points[ind[0][2]]

        n = np.cross(p1 - p, p2 - p)
        n = n / np.sqrt(np.sum(n ** 2))

        points.append(list(n))
    return points