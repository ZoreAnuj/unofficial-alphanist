import math
from collections.abc import Collection
from typing import Literal

import numba as nb
import numpy as np

_F64_MATRIX = nb.types.Array(nb.types.float64, 2, "C")
_F64_VECTOR = nb.types.Array(nb.types.float64, 1, "C")
# Default ATOL, same as numpy. Uses a reasonable angle atol separate from position
DEFAULT_ATOL = 1e-8
DEFAULT_ANGLE_ATOL = 1e-3
# Half-width (radians) of the band around theta = pi where log_SO3 switches from
# the antisymmetric axis formula to the symmetric-part recovery. NOT an accuracy
# bound: it is the crossover between two methods whose errors grow in opposite
# directions — the antisymmetric formula degrades as ~eps/(pi-theta)^2 toward pi,
# the symmetric recovery as ~(pi-theta)^2 away from it. Their curves cross near
# pi-theta = 1e-4 (both ~1e-8), which is where the seam belongs; the resulting
# worst-case rotation-vector error stays ~1e-8 across the whole range.
_LOG_SO3_NEAR_PI_TOL = 1e-4


def _numba_can_cache() -> bool:
    """Whether numba finds a writable cache location for this file.

    cache=True raises "no locator available" at decoration when NUMBA_CACHE_DIR
    is unset and neither this directory's __pycache__ nor the user cache dir
    ($HOME/.cache) is writable, e.g. a container running as an arbitrary uid
    with HOME=/. Compile uncached there instead of failing every import.
    """
    try:
        nb.njit(cache=True)(lambda: None)
    except RuntimeError as exc:
        if "no locator available" not in str(exc):
            raise
        return False
    return True


_NUMBA_CACHE = _numba_can_cache()


def sign(val: float) -> Literal[-1, 0, 1]:
    """The sign of the value. Similar but faster to np or math sign."""
    if val > 0.0:
        return 1
    elif val < 0.0:
        return -1
    else:
        return 0


def linear_ramp(val: float, width: float = 1) -> float:
    """A ramped value by the width between 0 and 1."""
    return clip(val / width, 0.0, 1.0)


def clip(val: float, lower: float, upper: float) -> float:
    """Clips the value between the lower and upper bounds."""
    if val < lower:
        return lower
    elif val > upper:
        return upper
    else:
        return val


def mean(values: Collection[float]) -> float:
    """The mean of the values."""
    return sum(values) / len(values)


def wrap_degrees(angle: float) -> float:
    """Wraps the angle to the range [-180, 180] degrees."""
    return ((angle + 180) % 360) - 180


@nb.njit(nb.float64(nb.float64), cache=_NUMBA_CACHE)
def _wrap_radian(angle: float) -> float:
    """Wraps an angle to the range [-pi, pi] radians."""
    return (angle + math.pi) % (2 * math.pi) - math.pi  # Wraps to [-pi, pi]


@nb.njit(nb.float64(_F64_VECTOR), cache=_NUMBA_CACHE)
def norm_3d_vector(vector: np.ndarray) -> float:
    """Computes the Euclidean norm of a 3D vector. Equivalent to np.linalg.norm."""
    return math.sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
    )


@nb.njit(_F64_MATRIX(nb.float64, nb.float64, nb.float64), cache=_NUMBA_CACHE)
def extrinsic_xyz_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """A much faster version of a sst.Rotation.from_euler("xyz", angles).as_matrix()
    The input angles of [roll, pitch, yaw] match to a rotation matrix defined as
    R = R_z(yaw) R_y(pitch) R_x(roll)

    The notation follows the reference:
    https://ntrs.nasa.gov/citations/19770019231 page 22
    The paper gives ZYX (intrinsic) which is equal XYZ (extrinsic).
    """
    rot = np.zeros((3, 3))

    c1 = np.cos(yaw)
    c2 = np.cos(pitch)
    c3 = np.cos(roll)

    s1 = np.sin(yaw)
    s2 = np.sin(pitch)
    s3 = np.sin(roll)

    # C order
    rot[0, 0] = c1 * c2
    rot[0, 1] = c1 * s2 * s3 - s1 * c3
    rot[0, 2] = c1 * s2 * c3 + s1 * s3

    rot[1, 0] = s1 * c2
    rot[1, 1] = s1 * s2 * s3 + c1 * c3
    rot[1, 2] = s1 * s2 * c3 - c1 * s3

    rot[2, 0] = -s2
    rot[2, 1] = c2 * s3
    rot[2, 2] = c2 * c3

    return rot


@nb.njit(_F64_MATRIX(nb.float64, nb.float64, nb.float64), cache=_NUMBA_CACHE)
def intrinsic_xyz_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """A much faster version of an XYZ intrinsic rotation such as
    sst.Rotation.from_euler("zyx", angles).as_matrix(). The input angles of
    [roll, pitch, yaw] match to a rotation matrix defined as
    R = R_x(roll) R_y(pitch) R_z(yaw)

    The notation follows the reference:
    https://ntrs.nasa.gov/citations/19770019231 page 13 which gives XYZ (intrinsic).
    """
    rot = np.zeros((3, 3))

    c1 = np.cos(roll)
    c2 = np.cos(pitch)
    c3 = np.cos(yaw)

    s1 = np.sin(roll)
    s2 = np.sin(pitch)
    s3 = np.sin(yaw)

    # C order
    rot[0, 0] = c2 * c3
    rot[0, 1] = -c2 * s3
    rot[0, 2] = s2

    rot[1, 0] = s1 * s2 * c3 + c1 * s3
    rot[1, 1] = -s1 * s2 * s3 + c1 * c3
    rot[1, 2] = -s1 * c2

    rot[2, 0] = -c1 * s2 * c3 + s1 * s3
    rot[2, 1] = c1 * s2 * s3 + s1 * c3
    rot[2, 2] = c1 * c2

    return rot


@nb.njit(_F64_MATRIX(nb.float64), cache=_NUMBA_CACHE)
def x_axis_transform(angle: float) -> np.ndarray:
    """A 4x4 homogeneous transform matrix representation of a rigid body configuration in
    SE3 (Special Euclidean group). Rotates about the x-axis by the angle specified. Can
    be multiplied by other transforms to represent a frame configuration.

    Note:
    https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
    https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    """
    transform = np.eye(4)
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    transform[1, 1] = cos_val
    transform[1, 2] = -sin_val

    transform[2, 1] = sin_val
    transform[2, 2] = cos_val

    return transform


@nb.njit(_F64_MATRIX(nb.float64), cache=_NUMBA_CACHE)
def y_axis_transform(angle: float) -> np.ndarray:
    """A 4x4 homogeneous transform matrix representation of a rigid body configuration in
    SE3 (Special Euclidean group). Rotates about the y-axis by the angle specified. Can
    be multiplied by other transforms to represent a frame configuration.

    Note:
    https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
    https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    """
    transform = np.eye(4)
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    transform[0, 0] = cos_val
    transform[2, 0] = -sin_val

    transform[0, 2] = sin_val
    transform[2, 2] = cos_val

    return transform


@nb.njit(_F64_MATRIX(nb.float64), cache=_NUMBA_CACHE)
def z_axis_transform(angle: float) -> np.ndarray:
    """A 4x4 homogeneous transform matrix representation of a rigid body configuration in
    SE3 (Special Euclidean group). Rotates about the z-axis by the angle specified. Can
    be multiplied by other transforms to represent a frame configuration.

    Note:
    https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
    https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    """
    transform = np.eye(4)
    cos_val = np.cos(angle)
    sin_val = np.sin(angle)
    transform[0, 0] = cos_val
    transform[1, 0] = sin_val

    transform[0, 1] = -sin_val
    transform[1, 1] = cos_val
    return transform


def transpose(matrix: np.ndarray) -> np.ndarray:
    """Returns the transpose of a matrix as a contiguous array."""
    return np.ascontiguousarray(matrix.T)


@nb.njit(_F64_MATRIX(_F64_VECTOR), cache=_NUMBA_CACHE)
def translation_transform(coord: np.ndarray) -> np.ndarray:
    """A 4x4 homogeneous transform matrix representation of a rigid body configuration in
    SE3 (Special Euclidean group). Translates by the vector specified. Can be
    multiplied by other transforms to represent a frame configuration.

    Note:
    https://faculty.sites.iastate.edu/jia/files/inline-files/homogeneous-transform.pdf
    https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/
    """
    rot = np.eye(4)
    rot[0, 3] = coord[0]
    rot[1, 3] = coord[1]
    rot[2, 3] = coord[2]
    return rot


@nb.njit(
    _F64_MATRIX(nb.float64, nb.float64, nb.float64, nb.float64), cache=_NUMBA_CACHE
)
def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """One link's 4x4 transform in standard (distal) Denavit-Hartenberg parameters.

    Equals the product Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha): theta and
    alpha in radians, d (offset along z) and a (length along x) in meters. Chaining
    these per joint gives a serial arm's forward kinematics. See Spong et al.,
    "Robot Modeling and Control" Sec. 3.1.2, or Lynch & Park, "Modern Robotics"
    Ch. 4 (Appendix on the DH parameterization).
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    transform = np.eye(4)
    transform[0, 0] = ct
    transform[0, 1] = -st * ca
    transform[0, 2] = st * sa
    transform[0, 3] = a * ct
    transform[1, 0] = st
    transform[1, 1] = ct * ca
    transform[1, 2] = -ct * sa
    transform[1, 3] = a * st
    transform[2, 1] = sa
    transform[2, 2] = ca
    transform[2, 3] = d
    return transform


@nb.njit(_F64_VECTOR(_F64_MATRIX), cache=_NUMBA_CACHE)
def VecSO3(w_hat: np.ndarray) -> np.ndarray:
    """The mapping from the Lie Algebra skew symmetric matrix so(3) to R3."""
    w = np.zeros(3)
    w[0] = w_hat[2, 1]
    w[1] = w_hat[0, 2]
    w[2] = w_hat[1, 0]
    return w


# Functions that need to be compiled before functions that depend it
@nb.njit(cache=_NUMBA_CACHE)
def is_close(a: float, b: float, atol: float = 1.0e-15) -> bool:
    """Checks if two floats are approximately equal within a tolerance."""
    return np.abs(a - b) < atol


@nb.njit(_F64_VECTOR(_F64_MATRIX), cache=_NUMBA_CACHE)
def log_SO3(rot: np.ndarray) -> np.ndarray:
    """The mapping SO(3) -> so(3) -> R3: a rotation matrix to its rotation vector.

    The result is the rotation axis scaled by the angle theta. The general branch
    reads the axis off the antisymmetric part of R; that part vanishes as
    theta -> pi, where R becomes symmetric and (R - R^T) is pure roundoff, so near
    pi the axis is instead recovered from the symmetric part, (R + I)/2 = w w^T,
    using the largest diagonal entry as the pivot for numerical conditioning.

    The three-branch structure (theta=0; symmetric-part recovery at theta=pi;
    antisymmetric formula otherwise) follows Lynch & Park, "Modern Robotics"
    (Cambridge, 2017), Sec. 3.2.3.3, a.k.a. MatrixLog3 in their reference code;
    see also Murray, Li & Sastry (1994), Sec. 2.3.
    - Book (free PDF): https://hades.mech.northwestern.edu/index.php/Modern_Robotics
    - MatrixLog3: https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py
    - SciPy oracle: scipy.spatial.transform.Rotation.as_rotvec

    Two robustness changes here go BEYOND the textbook, because its theta=pi
    branch is gated on the exact test tr(R) <= -1: a numerically perturbed pi
    rotation lands at tr(R) = -1 + eps and the textbook code (and MatrixLog3)
    falls through to the antisymmetric branch, dividing roundoff by sin(theta) ~ 0
    and collapsing the axis to ~0. So instead: (1) the symmetric branch is entered
    over a finite band (theta > pi - _LOG_SO3_NEAR_PI_TOL), not only at tr(R) <= -1;
    (2) the overall sign, which the textbook leaves arbitrary at pi (pi*w == -pi*w),
    is restored from the antisymmetric part so the band stays correct for theta
    just below pi.
    """
    # Clamp against float drift pushing the argument outside arccos's domain
    # (a valid rotation matrix can give a trace marginally outside [-1, 3]).
    cos_theta = (np.trace(rot) - 1.0) / 2.0
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)

    if is_close(theta, 0.0):
        return np.zeros(3)

    if theta > np.pi - _LOG_SO3_NEAR_PI_TOL:
        # (R + I)/2 = w w^T near theta = pi: diagonal gives w_i^2, off-diagonals
        # give the relative signs. Pivot on the largest |w_i| to avoid dividing
        # by a near-zero component.
        diag = np.empty(3)
        for i in range(3):
            half = (rot[i, i] + 1.0) / 2.0
            diag[i] = half if half > 0.0 else 0.0

        pivot = 0
        if diag[1] > diag[pivot]:
            pivot = 1
        if diag[2] > diag[pivot]:
            pivot = 2

        axis = np.empty(3)
        axis[pivot] = np.sqrt(diag[pivot])
        for j in range(3):
            if j != pivot:
                # (w w^T)_{pivot,j} = (R_{pivot,j} + R_{j,pivot}) / 4
                axis[j] = (rot[pivot, j] + rot[j, pivot]) / 4.0 / axis[pivot]
        axis = axis / norm_3d_vector(axis)

        # The symmetric part fixes the axis only up to an overall sign. The
        # antisymmetric part is proportional to sin(theta) * w, so its pivot
        # component still carries the correct sign for theta just below pi; at
        # exactly pi it is roundoff, where the sign is moot (pi*w == -pi*w).
        if pivot == 0:
            antisym_pivot = rot[2, 1] - rot[1, 2]
        elif pivot == 1:
            antisym_pivot = rot[0, 2] - rot[2, 0]
        else:
            antisym_pivot = rot[1, 0] - rot[0, 1]
        if antisym_pivot < 0.0:
            axis = -axis

        return axis * theta

    w_hat = (rot - np.transpose(rot)) / (2.0 * np.sin(theta)) * theta
    return VecSO3(w_hat)


@nb.njit(_F64_VECTOR(_F64_MATRIX, _F64_MATRIX), cache=_NUMBA_CACHE)
def residual_SO3(r_1: np.ndarray, r_2: np.ndarray) -> np.ndarray:
    """Computes the rotation vector that rotates R_2 to R_1.

    Derived from R_1 = R_2 exp_SO3(w), where exp_SO3 is inverse mapping of log_SO3.
    """
    return log_SO3(np.dot(np.transpose(r_2), r_1))


@nb.njit(_F64_VECTOR(_F64_MATRIX, _F64_MATRIX), cache=_NUMBA_CACHE)
def residual_SE3(pose_se3: np.ndarray, target_se3: np.ndarray) -> np.ndarray:
    """Computes the difference between two poses in SE3 = (SO3, R3)."""
    r = np.zeros(6)

    # Slices of arrays are not contiguous arrays
    p_t, R_t = target_se3[:3, 3], np.ascontiguousarray(target_se3[:3, :3])
    p_fk, R_fk = pose_se3[:3, 3], np.ascontiguousarray(pose_se3[:3, :3])

    r[:3] = p_fk - p_t
    r[3:] = residual_SO3(R_fk, R_t)

    return r


@nb.njit(_F64_VECTOR(_F64_MATRIX), cache=_NUMBA_CACHE)
def as_euler_xyz(rot: np.ndarray) -> np.ndarray:
    """Converts a rotation matrix to extrinsic XYZ Euler angles [roll, pitch, yaw] in radians."""
    # Should never be in gimbal lock so no need to define.
    # Clamp to [-1, 1] to guard against floating-point drift producing NaN from asin.
    sp = -rot[2, 0]
    if sp > 1.0:
        sp = 1.0
    elif sp < -1.0:
        sp = -1.0
    phi = _wrap_radian(math.atan2(rot[2, 1], rot[2, 2]))
    theta = _wrap_radian(math.asin(sp))
    psi = _wrap_radian(math.atan2(rot[1, 0], rot[0, 0]))
    return np.array([phi, theta, psi])


@nb.njit(_F64_VECTOR(nb.float64, nb.float64), cache=_NUMBA_CACHE)
def celestial_coordinates(altitude: float, azimuth: float) -> np.ndarray:
    """Converts celestial coordinates (altitude, azimuth) to Cartesian (x, y, z)."""
    alt_cos = math.cos(altitude)

    x = alt_cos * math.cos(azimuth)
    y = alt_cos * math.sin(azimuth)
    z = math.sin(altitude)

    return np.array([x, y, z])


@nb.njit(_F64_VECTOR(nb.float64, nb.float64, nb.float64), cache=_NUMBA_CACHE)
def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Converts Euler angles (roll, pitch, yaw) in degrees to quaternion (w, x, y, z)."""
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


@nb.njit(
    _F64_VECTOR(nb.float64, nb.float64, nb.float64, nb.float64), cache=_NUMBA_CACHE
)
def quaternion_to_euler(w: float, x: float, y: float, z: float) -> np.ndarray:
    """A conversion from quaternion to euler RPY (roll, pitch, yaw) in degrees.
    Normalizes the quaternion for you before performing the conversion.
    """
    norm = (w**2 + x**2 + y**2 + z**2) ** 0.5
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    # Roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # Pitch (y-axis rotation)
    # TODO: Validate that sqrt args are clamped to >= 0. For non-unit or
    # numerically imprecise quaternions the argument can go negative, raising
    # ValueError.
    sinp = math.sqrt(1 + 2 * (w * y - x * z))
    cosp = math.sqrt(1 - 2 * (w * y - x * z))
    pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2

    # Yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return np.array([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])


@nb.njit(_F64_VECTOR(_F64_VECTOR, _F64_VECTOR, nb.float64), cache=_NUMBA_CACHE)
def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (w, x, y, z)."""
    dot = np.dot(q0, q1)

    # Ensure shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = min(dot, 1.0)

    if dot > 0.9995:
        # Very close quaternions — linear interpolation
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * q0 + s1 * q1


@nb.njit(
    nb.boolean(nb.types.UniTuple(nb.float64, 2), nb.float64[:, :]), cache=_NUMBA_CACHE
)
def polygon_contains(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """A raycasting algorithm to determine if the point is contained within the polygon
    represented as a numpy array with polygon points.
    """
    px, py = point
    intersection_count = 0
    n = polygon.shape[0]

    for i in range(n):
        x1, y1 = polygon[i]
        # Ensure the last edge connects to the first point
        x2, y2 = polygon[(i + 1) % n]

        # Check if the point is between y1 and y2
        if (py > min(y1, y2)) and (py <= max(y1, y2)) and (px <= max(x1, x2)):
            # Compute the intersection point's x-coordinate. Avoids division by zero.
            x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1) if y1 != y2 else x1

            # Count intersection if the point is to the left of it
            if px < x_intersect:
                intersection_count += 1

    return intersection_count % 2 == 1


@nb.njit(
    _F64_MATRIX(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
    cache=_NUMBA_CACHE,
)
def create_rpy_transform(
    x: float, y: float, z: float, roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix from xyz translation and roll,
    pitch, yaw rotations (degrees). First applies the rotations (roll around X, pitch
    around Y, yaw around Z) and then applies the translation.
    """
    # Create rotation matrix using extrinsic XYZ convention (same as roll, pitch, yaw)
    rot = extrinsic_xyz_rotation_matrix(
        math.radians(roll), math.radians(pitch), math.radians(yaw)
    )

    # Create homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = z

    return transform


@nb.njit(
    _F64_MATRIX(
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
    ),
    cache=_NUMBA_CACHE,
)
def create_rotary_joint_transform(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
    axis_x: float,
    axis_y: float,
    axis_z: float,
    joint_value: float,
) -> np.ndarray:
    """Creates a 4x4 transformation matrix for a rotary joint."""
    origin_transform = create_rpy_transform(x, y, z, roll, pitch, yaw)

    axis_norm = math.sqrt(axis_x**2 + axis_y**2 + axis_z**2)
    if axis_norm < 1e-10:  # Avoid division by zero
        return origin_transform

    axis_x /= axis_norm
    axis_y /= axis_norm
    axis_z /= axis_norm

    # Fill in rotation matrix
    # Create transformation matrix for joint movement
    joint_transform = np.eye(4)
    # Create rotation matrix around specified axis using Rodrigues' formula
    c = math.cos(joint_value)
    s = math.sin(joint_value)
    t = 1.0 - c

    joint_transform[0, 0] = t * axis_x * axis_x + c
    joint_transform[0, 1] = t * axis_x * axis_y - s * axis_z
    joint_transform[0, 2] = t * axis_x * axis_z + s * axis_y

    joint_transform[1, 0] = t * axis_x * axis_y + s * axis_z
    joint_transform[1, 1] = t * axis_y * axis_y + c
    joint_transform[1, 2] = t * axis_y * axis_z - s * axis_x

    joint_transform[2, 0] = t * axis_x * axis_z - s * axis_y
    joint_transform[2, 1] = t * axis_y * axis_z + s * axis_x
    joint_transform[2, 2] = t * axis_z * axis_z + c

    # Combine transformations: origin transform * joint transform
    # For a joint, we first apply the origin transformation (fixing the joint in space),
    # then apply the joint movement transform
    return np.dot(origin_transform, joint_transform)


@nb.njit(_F64_MATRIX(_F64_VECTOR), cache=_NUMBA_CACHE)
def poseuler_to_hmat(poseuler: np.ndarray) -> np.ndarray:
    """Convert a 6D pose [x, y, z, roll_rad, pitch_rad, yaw_rad] to a 4x4
    homogeneous transformation matrix using extrinsic XYZ Euler angles."""
    hmat = np.eye(4)
    hmat[:3, :3] = extrinsic_xyz_rotation_matrix(poseuler[3], poseuler[4], poseuler[5])
    hmat[:3, 3] = poseuler[:3]
    return hmat


@nb.njit(_F64_VECTOR(_F64_MATRIX), cache=_NUMBA_CACHE)
def hmat_to_poseuler(hmat: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transformation matrix to a 6D pose
    [x, y, z, roll_rad, pitch_rad, yaw_rad] using extrinsic XYZ Euler angles."""
    rot = hmat[:3, :3]
    sp = -rot[2, 0]
    if sp > 1.0:
        sp = 1.0
    elif sp < -1.0:
        sp = -1.0
    pitch = math.asin(sp)
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    result = np.empty(6)
    result[0] = hmat[0, 3]
    result[1] = hmat[1, 3]
    result[2] = hmat[2, 3]
    result[3] = roll
    result[4] = pitch
    result[5] = yaw
    return result


@nb.njit(_F64_MATRIX(_F64_MATRIX), cache=_NUMBA_CACHE)
def hmat_inv(hmat: np.ndarray) -> np.ndarray:
    """Numerically stable inverse of a 4x4 homogeneous transformation matrix.

    Exploits the orthogonality of the rotation block: R^-1 = R^T.
    """
    rot = hmat[:3, :3]
    pos = hmat[:3, 3]
    hinv = np.eye(4)
    hinv[:3, :3] = rot.T
    # t_inv = -R^T @ t
    hinv[0, 3] = -(rot[0, 0] * pos[0] + rot[1, 0] * pos[1] + rot[2, 0] * pos[2])
    hinv[1, 3] = -(rot[0, 1] * pos[0] + rot[1, 1] * pos[1] + rot[2, 1] * pos[2])
    hinv[2, 3] = -(rot[0, 2] * pos[0] + rot[1, 2] * pos[1] + rot[2, 2] * pos[2])
    return hinv


@nb.njit(_F64_MATRIX(_F64_VECTOR, _F64_VECTOR), cache=_NUMBA_CACHE)
def pos_quat_to_hmat(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Convert position [x,y,z] and quaternion [w,i,j,k] to a 4x4 hmat."""
    q = quat.copy()
    nq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
    if nq < 1e-10:
        return np.eye(4)
    s = math.sqrt(2.0 / nq)
    q = q * s

    # Outer product elements (q00 not needed in the rotation formula)
    q01 = q[0] * q[1]
    q02 = q[0] * q[2]
    q03 = q[0] * q[3]
    q11 = q[1] * q[1]
    q12 = q[1] * q[2]
    q13 = q[1] * q[3]
    q22 = q[2] * q[2]
    q23 = q[2] * q[3]
    q33 = q[3] * q[3]

    hmat = np.eye(4)
    hmat[0, 0] = 1.0 - q22 - q33
    hmat[0, 1] = q12 - q03
    hmat[0, 2] = q13 + q02
    hmat[1, 0] = q12 + q03
    hmat[1, 1] = 1.0 - q11 - q33
    hmat[1, 2] = q23 - q01
    hmat[2, 0] = q13 - q02
    hmat[2, 1] = q23 + q01
    hmat[2, 2] = 1.0 - q11 - q22
    hmat[:3, 3] = pos
    return hmat


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Creates a skew-symmetric matrix from a 3D vector for cross product operation."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


@nb.njit(
    _F64_VECTOR(_F64_VECTOR, _F64_MATRIX, _F64_VECTOR, _F64_VECTOR), cache=_NUMBA_CACHE
)
def closest_point_on_box(
    box_pos: np.ndarray, box_rot: np.ndarray, box_half: np.ndarray, point: np.ndarray
) -> np.ndarray:
    """Return the closest point on an oriented box's surface to a world-space point."""
    local = box_rot.T @ (point - box_pos)
    for i in range(3):
        if local[i] < -box_half[i]:
            local[i] = -box_half[i]
        elif local[i] > box_half[i]:
            local[i] = box_half[i]
    return box_pos + box_rot @ local


@nb.njit(
    nb.float64(
        _F64_VECTOR, _F64_MATRIX, _F64_VECTOR, _F64_VECTOR, _F64_MATRIX, _F64_VECTOR
    ),
    cache=_NUMBA_CACHE,
)
def box_to_box_distance(
    pos_a: np.ndarray,
    rot_a: np.ndarray,
    half_a: np.ndarray,
    pos_b: np.ndarray,
    rot_b: np.ndarray,
    half_b: np.ndarray,
) -> float:
    """Compute the minimum distance between two oriented bounding boxes.

    Uses iterative closest-point projection between two convex OBBs. Each
    iteration projects a point onto one box, then the result onto the other.
    Convergence is guaranteed for convex shapes within a few iterations.
    """
    p_a = pos_a.copy()
    p_b = pos_b.copy()
    # 3 is sufficient to get the closest points on the box
    for _ in range(3):
        p_a = closest_point_on_box(pos_a, rot_a, half_a, p_b)
        p_b = closest_point_on_box(pos_b, rot_b, half_b, p_a)

    diff = p_a - p_b
    return np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
