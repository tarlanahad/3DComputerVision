import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def pts2df(pts):
    return pd.DataFrame(pts, columns=['x', 'y', 'z'])


def ply2df(path):
    plydata = PlyData.read(path)

    # Extract the vertex coordinates
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    return pts2df(np.column_stack((x, y, z)))


def rotate_point_cloud(vertex_data, rotation, azimuth):
    # Convert rotation and azimuth to radians
    rotation = np.radians(rotation)
    azimuth = np.radians(azimuth)

    # Define the rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotation), -np.sin(rotation)],
                    [0, np.sin(rotation), np.cos(rotation)]])
    R_y = np.array([[np.cos(azimuth), 0, np.sin(azimuth)],
                    [0, 1, 0],
                    [-np.sin(azimuth), 0, np.cos(azimuth)]])

    # Rotate the point cloud
    rotated_vertex_data = np.dot(vertex_data, R_y.T)
    rotated_vertex_data = np.dot(rotated_vertex_data, R_x.T)

    return pts2df(rotated_vertex_data)


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


def add_noise_rotate_translate_pts(vertex_data, noise_std=0.01, translation_range=10):
    noisy_vertex_data = vertex_data + np.random.normal(0, noise_std, vertex_data.shape)

    rotation_vector = (2 * np.random.rand(3) - 1) * 2 * np.pi
    rotated_vertex_data = so3_rotation(noisy_vertex_data, rotation_vector)

    translation_vector = [np.random.uniform(-translation_range, translation_range) for _ in range(3)]

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

    return original_pts, R, t


def trimmed_icp(original_pts, modified_pts, max_iterations=50, tolerance=1e-10, threshold=0.1):
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

    return original_pts, R, t


def quaternion_estimate_transformation(source_points, target_points):
    # Estimate the centroids of the point clouds
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    # Subtract the centroids from the point clouds
    centered_source_points = source_points - source_centroid
    centered_target_points = target_points - target_centroid

    print(centered_target_points.shape, centered_source_points.shape)
    # Compute the cross-covariance matrix
    H = centered_target_points.T @ centered_source_points

    # Compute the quaternion that maximizes the trace of H
    quat = np.linalg.eig(H)[1][:, np.argmax(np.linalg.eig(H)[0])]

    # Normalize the quaternion
    quat /= np.linalg.norm(quat)

    # Compute the rotation matrix from the quaternion
    R = quat_to_rotation_matrix(quat)

    # Compute the translation from the centroids
    t = target_centroid - R @ source_centroid

    # Return the transformation as a 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quat_to_rotation_matrix(q):
    # Compute the rotation matrix from the quaternion
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
