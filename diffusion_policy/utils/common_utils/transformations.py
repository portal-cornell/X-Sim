import cv2
import numpy as np
from typing import Any, List, Tuple, Dict, Union
from scipy.spatial.transform import Rotation as R


def triangulate_points(
    pixels1: np.ndarray,
    pixels2: np.ndarray,
    intrinsics: Dict[str, np.ndarray],
    transforms: Dict[str, Dict[str, np.ndarray]],
) -> np.ndarray:
    """
    Triangulates 3D points from two sets of 2D pixels using the camera intrinsics and extrinsics.

    Args:
        pixels1: The first set of 2D points. Can be either of shape (num_points, 2) or (action_horizon, num_points, 2).
        pixels2: The second set of 2D points. Can be either of shape (num_points, 2) or (action_horizon, num_points, 2).
        intrinsics: A dictionary containing the intrinsic matrices of the cameras.
        transforms: A dictionary containing the transformation matrices of the cameras.

    Returns:
        A numpy array of 3D points.
    """
    P1 = intrinsics["agent1"] @ transforms["agent1"]["trc"]
    P2 = intrinsics["agent2"] @ transforms["agent2"]["trc"]

    def triangulate(pts1, pts2):
        points1_h = np.vstack((pts1.T, np.ones((1, pts1.shape[0]))))
        points2_h = np.vstack((pts2.T, np.ones((1, pts2.shape[0]))))
        points_4d_homog = cv2.triangulatePoints(P1, P2, points1_h[:2], points2_h[:2])
        points_3d_homog = points_4d_homog / points_4d_homog[3]
        return points_3d_homog[:3].T

    if pixels1.ndim == 3 and pixels2.ndim == 3:
        points_3d = [triangulate(pts1, pts2) for pts1, pts2 in zip(pixels1, pixels2)]
        # return as (action_horizon, num_points, 3)
        return np.array(points_3d)
    else:
        # return as (num_points, 3)
        return triangulate(pixels1, pixels2)


def update_gripper_transform(
    curr_gripper_mesh: Any,
    future_gripper_mesh: Any,
    proprio: np.ndarray,
    points_3d_dt: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the transformation of the current and future gripper meshes based on the provided proprioceptive data and 3D points,
    and returns the transformation matrices for both.

    Args:
        curr_gripper_mesh (Any): The mesh object of the current gripper state.
        future_gripper_mesh (Any): The mesh object of the future gripper state.
        proprio (np.ndarray): The proprioceptive data array, containing the current end-effector position and quaternion.
        points_3d_dt (List[np.ndarray]): A list of 3D points at the current and future time steps.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the transformation matrices for the current and future gripper meshes.
    """
    # Calculate rigid transformation between current and future points
    curr_points = points_3d_dt[0]
    future_points = points_3d_dt[-1]
    transformation_matrix = solve_for_rigid_transformation(curr_points, future_points)

    # Update gripper transforms + visualize
    curr_ee_pos = proprio[:3]
    curr_ee_eul = proprio[3:6]
    curr_ee_quat = R.from_euler("xyz", curr_ee_eul).as_quat()
    curr_gripper_transform = get_transform_from_pos_quat(curr_ee_pos, curr_ee_quat)
    future_gripper_transform = np.dot(transformation_matrix, curr_gripper_transform)
    future_ee_pos = future_gripper_transform[:3, 3]
    future_ee_rot = future_gripper_transform[:3, :3]
    future_ee_quat = R.from_matrix(future_ee_rot).as_quat()
    future_gripper_transform = get_transform_from_pos_quat(
        future_ee_pos, future_ee_quat
    )
    curr_gripper_mesh.transform(curr_gripper_transform)
    future_gripper_mesh.transform(future_gripper_transform)

    return curr_gripper_transform, future_gripper_transform


def solve_for_rigid_transformation(inpts: np.ndarray, outpts: np.ndarray) -> np.ndarray:
    """
    Computes the rigid transformation matrix that aligns one set of points to another.
    Uses the Kabsch algorithm to solve for the optimal rotation and translation.

    Args:
        inpts (np.ndarray): The source points as a NxD array, where N is the number of points and D is the dimension.
        outpts (np.ndarray): The destination points as a NxD array, matching the source points.

    Returns:
        np.ndarray: A 3x4 rigid transformation matrix that aligns the source points to the destination points.
    """
    assert (
        inpts.shape == outpts.shape
    ), "Input and output points must have the same shape."
    inpts, outpts = np.copy(inpts), np.copy(outpts)
    inpt_mean = inpts.mean(axis=0)
    outpt_mean = outpts.mean(axis=0)
    outpts -= outpt_mean
    inpts -= inpt_mean
    X = inpts.T
    Y = outpts.T
    covariance = np.dot(X, Y.T)
    U, s, V = np.linalg.svd(covariance)
    S = np.diag(s)
    assert np.allclose(covariance, np.dot(U, np.dot(S, V)))
    V = V.T
    idmatrix = np.identity(3)
    idmatrix[2, 2] = np.linalg.det(np.dot(V, U.T))
    R = np.dot(np.dot(V, idmatrix), U.T)
    t = outpt_mean.T - np.dot(R, inpt_mean.T)
    T = np.zeros((3, 4))
    T[:3, :3] = R
    T[:, 3] = t
    return T


def get_transform_from_pos_quat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Constructs a 4x4 transformation matrix from position and quaternion.

    Args:
        pos (np.ndarray): The position vector as a 3-element array.
        quat (np.ndarray): The quaternion as a 4-element array (x, y, z, w).

    Returns:
        np.ndarray: A 4x4 transformation matrix.
    """
    pose_transform = np.eye(4)
    rotation_matrix = R.from_quat(quat).as_matrix()
    pose_transform[:3, :3] = rotation_matrix
    pose_transform[:3, 3] = pos
    return pose_transform
