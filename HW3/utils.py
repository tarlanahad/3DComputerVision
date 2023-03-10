import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from timeit import default_timer as timer


def pts2df(pts):
    return pd.DataFrame(pts, columns=['x', 'y', 'z'])


def ply2df(path):
    plydata = PlyData.read(path)

    # Extract the vertex coordinates
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    return pts2df(np.column_stack((x, y, z)))


def get_farthest_distance(points):
    pca = PCA(n_components=1)
    reduced_points = pca.fit_transform(points)
    return reduced_points.max() - reduced_points.min()


def find_angular_error(pc1, pc2):
    pc1_mean = np.mean(pc1, axis=0)
    pc2_mean = np.mean(pc2, axis=0)
    pc1_centered = pc1 - pc1_mean
    pc2_centered = pc2 - pc2_mean

    # Compute the covariance matrix
    cov = np.dot(pc1_centered.T, pc2_centered)

    U, S, Vt = np.linalg.svd(cov)

    r = R.from_matrix(np.dot(U, Vt))
    angular_error = r.as_euler('xyz', degrees=True)
    return angular_error


def rotate_point_cloud(angle_x, angle_y, angle_z):
    x = np.radians(angle_x)
    y = np.radians(angle_y)
    z = np.radians(angle_z)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])

    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])

    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def so3_rotation(vertex_data, rotation_vector):
    theta = np.linalg.norm(rotation_vector)
    k = rotation_vector / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    rotated_vertex_data = np.dot(vertex_data, R.T)
    return pts2df(rotated_vertex_data)


def rotation_matrix_to_rotation_vector(R):
    # Compute the rotation angle
    theta = np.arccos((np.trace(R) - 1) / 2)

    n = 1 / (2 * np.sin(theta)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    # Compute the rotation vector
    rotation_vector = theta * n
    return -rotation_vector


def add_noise_rotate_translate_pts(vertex_data, noise_std=0.01, degrees=[45, 45, 45],
                                   translation_vector=np.array([5, 5, 5])):
    noisy_vertex_data = vertex_data + np.random.normal(0, noise_std, vertex_data.shape)

    R = rotate_point_cloud(degrees[0], degrees[1], degrees[2])

    rotation_vector = rotation_matrix_to_rotation_vector(R)
    rotated_vertex_data = so3_rotation(noisy_vertex_data, rotation_vector)

    translated_rotated_vertex_data = rotated_vertex_data + translation_vector

    return pts2df(translated_rotated_vertex_data), translation_vector, rotation_vector


# Metrices #
def mse(A, B):
    return ((A - B) ** 2).mean()


def rmse(A, B):
    return np.sqrt(mse(A, B))


def svd_rotation_estimation(original_pts, modified_pts):
    centered_modified_pts = modified_pts - np.mean(modified_pts, axis=0)
    centered_original_pts = original_pts - np.mean(original_pts, axis=0)

    U, _, Vh = np.linalg.svd(centered_original_pts.T @ centered_modified_pts)
    R = np.dot(U, Vh)

    return R


def estimate_translation(modified_pts, original_pts):
    return np.array(np.mean(modified_pts, axis=0) - np.mean(original_pts, axis=0))


def nearest_neighbor(original_pts, modified_pts, kdtree=None):
    if kdtree is None:
        kdtree = cKDTree(original_pts)
    distances, indices = kdtree.query(modified_pts)
    return distances, indices


def icp(original_pts, modified_pts, max_iterations=50, tolerance=1e-10):
    start = timer()
    original_pts = np.array(original_pts)
    modified_pts = np.array(modified_pts)
    e = 1e9
    tree = None
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(modified_pts, original_pts, tree)

        t = estimate_translation(modified_pts, original_pts[indices])
        R = svd_rotation_estimation(original_pts, modified_pts)

        original_pts = original_pts @ R + t

        prev_e = e
        e = rmse(original_pts, modified_pts)
        if np.abs(e - prev_e) < tolerance:
            break

    return original_pts, i, timer() - start


def trimmed_icp(original_pts, modified_pts, max_iterations=50, tolerance=1e-10, threshold=0.1):
    start = timer()
    original_pts = np.array(original_pts)
    modified_pts = np.array(modified_pts)
    e = 1e9
    tree = None
    for i in range(max_iterations):
        distances, indices = nearest_neighbor(modified_pts, original_pts, tree)

        indices = indices[distances < threshold]

        t = estimate_translation(modified_pts, original_pts[indices])
        R = svd_rotation_estimation(original_pts, modified_pts)

        original_pts = original_pts @ R + t

        prev_e = e
        e = rmse(original_pts, modified_pts)
        if np.abs(e - prev_e) < tolerance:
            break

    return original_pts, i, timer() - start


'''
------- QUATERNION -----------
Find the centroid of both point clouds.
Subtract the centroid from all points in both point clouds.
Compute the correlation matrix H between the two point clouds.
Compute the quaternion that represents the rotation by taking the eigenvector of H corresponding to the largest eigenvalue.
Convert the quaternion to a rotation matrix.
'''
