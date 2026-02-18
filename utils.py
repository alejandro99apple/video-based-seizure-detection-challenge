import numpy as np
import pandas as pd
from antropy import sample_entropy
from scipy.stats import kurtosis


def fill_nan_with_last_valid(lmk_arr):
    """
    Fill NaNs while maintaining the last known position of the marker.
    This avoids 'hallucinating' linear trajectories and preserves velocity peaks.
    """
    # lmk_arr shape: (150, 33, 5)
    for l in range(lmk_arr.shape[1]): # 33 markers
        for c in range(lmk_arr.shape[2]): # x, y, z, vis, pres
            series = lmk_arr[:, l, c]
            mask = np.isnan(series)
            
            if mask.any() and not mask.all():
                # Get indices of values that are NOT NaN
                idx = np.where(~mask, np.arange(len(mask)), 0)
                # Carry the last valid index forward
                np.maximum.accumulate(idx, out=idx)
                # Replace NaNs with the value at the last seen valid index
                series[mask] = series[idx[mask]]
                
                # Edge case: if the video starts with NaNs, fill with the first valid value
                remaining_nans = np.isnan(series)
                if remaining_nans.any():
                    first_val = series[~remaining_nans][0]
                    series[remaining_nans] = first_val
                    
            elif mask.all():
                series[mask] = 0.0 # If the marker never appeared in the video
                
    return lmk_arr

def preprocess_and_augment(lmk_arr, utility_threshold=0.5):
    """
    Keeps your original structure but changes the fill technique
    to avoid degrading the F1-score.
    """
    total_frames = lmk_arr.shape[0]
    
    # 1. Compute missing_ratio
    nan_per_frame = np.any(np.isnan(lmk_arr), axis=(1, 2))
    missing_ratio = np.sum(nan_per_frame) / total_frames
    
    # 2. Fill by 'freezing' (Last Observation Carried Forward)
    # Only if the video is useful enough
    if missing_ratio < utility_threshold:
        # IMPORTANT: Replace interpolation with fill_nan_with_last_valid
        lmk_arr_clean = fill_nan_with_last_valid(lmk_arr.copy())
    else:
        # If it is pure noise, keep zeros
        lmk_arr_clean = np.zeros_like(lmk_arr)

    # 3. Add missing_ratio channel (6th dimension)
    ratio_channel = np.full((total_frames, lmk_arr.shape[1], 1), missing_ratio)
    augmented_arr = np.concatenate([lmk_arr_clean, ratio_channel], axis=2)
    
    return augmented_arr


def compute_metrics(lmk_df, metrics:dict):

    for col in lmk_df.columns:
        signal = lmk_df[col].values.astype(float)

        # -------- Basic statistics --------
        metrics[f"{col}_mean"] = np.mean(signal)
        metrics[f"{col}_std"] = np.std(signal)
        metrics[f"{col}_max"] = np.max(signal)
        metrics[f"{col}_min"] = np.min(signal)
        metrics[f"{col}_kurtosis"] = kurtosis(signal)
        metrics[f"{col}_entropy"] = sample_entropy(signal, order=2)


        # -------- Temporal dynamics (NEW) --------
        velocity = np.diff(signal)                  # length = T-1
        acceleration = np.diff(velocity)            # length = T-2

        metrics[f"{col}_max_velocity"] = np.max(np.abs(velocity))
        metrics[f"{col}_max_acceleration"] = np.max(np.abs(acceleration))

    return metrics


def compute_null_metrics():
    metrics_list = {}
    for col in ["right_hand", "left_hand", "right_knee", "left_knee","right_foot", "left_foot", "visibility", "presence"]:
        metrics_list[f"{col}_mean"] = 0.0
        metrics_list[f"{col}_std"] = 0.0
        metrics_list[f"{col}_max"] = 0.0
        metrics_list[f"{col}_min"] = 0.0
        metrics_list[f"{col}_kurtosis"] = 0.0
        metrics_list[f"{col}_entropy"] = 0.0
        metrics_list[f"{col}_max_velocity"] = 0.0
        metrics_list[f"{col}_max_acceleration"] = 0.0

    return metrics_list


def process_data(lmk_arr, ID, label):

    # Function that extracts x, y, z, vis coordinates for each landmark in a frame
    def _safe_point(frame, idx):
        """Return (x, y, z) or None if index invalid or contains NaNs."""
        if idx >= len(frame):
            return None
        pt = frame[idx]
        if pt is None or np.isnan(pt).any():
            return None
        # Ensure length >= 3
        if len(pt) < 3:
            return None
        return (float(pt[0]), float(pt[1]), float(pt[2]))
    

    # Function to compute line intersection in the XY plane
    def _line_intersection_2d(p1, p2, p3, p4):
        """Return (x, y, z) of line intersection in XY plane, z averaged from interpolations.

        Points are (x, y, z). If lines are parallel or any point is None, return None.
        """
        if None in (p1, p2, p3, p4):
            return None

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        x4, y4, z4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # Parallel or coincident lines in XY

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        xi = x1 + t * (x2 - x1)
        yi = y1 + t * (y2 - y1)

        zi1 = z1 + t * (z2 - z1)
        zi2 = z3 + u * (z4 - z3)
        zi = (zi1 + zi2) / 2.0

        return (xi, yi, zi)
    
    # Function to compute the angle between three 3D points
    def _compute_angle(a, b, c):
        """Compute angle ABC in 3D space with B as the vertex.
        Returns angle/360 for normalization. Returns NaN if any point is None.
        """
        if None in (a, b, c):
            return np.nan
        
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c
        
        # Vectors BA and BC
        ba = np.array([ax - bx, ay - by, az - bz])
        bc = np.array([cx - bx, cy - by, cz - bz])
        
        # Magnitudes
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return np.nan
        
        # Cosine of angle
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Angle in degrees
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        return angle_deg / 360.0
    
    # Function to compute Euclidean distance between two 3D points
    def _distance(a, b):
        """Euclidean distance between two 3D points. Returns NaN if any is None."""
        if a is None or b is None:
            return np.nan
        ax, ay, az = a
        bx, by, bz = b
        return float(np.linalg.norm([ax - bx, ay - by, az - bz]))
    

    # Function body
    rows = []

    # Extract missing_ratio from the augmented array (index 5 in last dimension)
    missing_ratio_val = lmk_arr[0, 0, 5]
    metrics = {"ID": ID, "class": label, "missing_ratio": missing_ratio_val}

    # Early return if all data was replaced with zeros (exceeded utility threshold)
    if np.all(lmk_arr[:, :, :5] == 0):
        null_metrics = compute_null_metrics()
        metrics.update(null_metrics)
        return metrics

    for frame in lmk_arr:
        p12 = _safe_point(frame, 12)
        p23 = _safe_point(frame, 23)
        p11 = _safe_point(frame, 11)
        p24 = _safe_point(frame, 24)

        ref = _line_intersection_2d(p12, p23, p11, p24)

        # Calculate shoulder distance as normalization reference
        shoulder_distance = _distance(p11, p12)

        # Calculate raw distances
        head = _distance(_safe_point(frame, 0), ref)
        right_hand = _distance(_safe_point(frame, 16), ref)
        left_hand = _distance(_safe_point(frame, 15), ref)
        right_knee = _distance(_safe_point(frame, 26), ref)
        left_knee = _distance(_safe_point(frame, 25), ref)
        right_foot = _distance(_safe_point(frame, 28), ref)
        left_foot = _distance(_safe_point(frame, 27), ref)
        right_arm_angle = _compute_angle(_safe_point(frame, 12), _safe_point(frame, 14), _safe_point(frame, 16))
        left_arm_angle = _compute_angle(_safe_point(frame, 11), _safe_point(frame, 13), _safe_point(frame, 15))
        right_leg_angle = _compute_angle(_safe_point(frame, 24), _safe_point(frame, 26), _safe_point(frame, 28))
        left_leg_angle = _compute_angle(_safe_point(frame, 23), _safe_point(frame, 25), _safe_point(frame, 27))


        # Normalize distances by shoulder distance
        if shoulder_distance is not None and not np.isnan(shoulder_distance) and shoulder_distance > 0:
            head = head / shoulder_distance if not np.isnan(head) else np.nan
            right_hand = right_hand / shoulder_distance if not np.isnan(right_hand) else np.nan
            left_hand = left_hand / shoulder_distance if not np.isnan(left_hand) else np.nan
            right_knee = right_knee / shoulder_distance if not np.isnan(right_knee) else np.nan
            left_knee = left_knee / shoulder_distance if not np.isnan(left_knee) else np.nan
            right_foot = right_foot / shoulder_distance if not np.isnan(right_foot) else np.nan
            left_foot = left_foot / shoulder_distance if not np.isnan(left_foot) else np.nan
            right_arm_angle = right_arm_angle if not np.isnan(right_arm_angle) else np.nan
            left_arm_angle = left_arm_angle if not np.isnan(left_arm_angle) else np.nan
            right_leg_angle = right_leg_angle if not np.isnan(right_leg_angle) else np.nan
            left_leg_angle = left_leg_angle if not np.isnan(left_leg_angle) else np.nan
        else:
            # If shoulder distance is invalid, set all distances to NaN
            head = right_hand = left_hand = right_knee = left_knee = right_foot = left_foot = right_arm_angle = left_arm_angle = right_leg_angle = left_leg_angle = np.nan


        # Extract visibility, presence, and missing_ratio for each landmark
        visibility = [
            (np.nan if (pt is None or len(pt) < 4 or (isinstance(pt[3], float) and np.isnan(pt[3])) ) else pt[3])
            for pt in frame
        ]
        presence = [
            (np.nan if (pt is None or len(pt) < 5 or (isinstance(pt[4], float) and np.isnan(pt[4])) ) else pt[4])
            for pt in frame
        ]

        # Compute per-frame min values (ignoring NaNs)
        visibility_mean = float(np.nanmin(visibility)) if len(visibility) else np.nan
        presence_mean = float(np.nanmin(presence)) if len(presence) else np.nan


        rows.append(
            {
                "right_hand": right_hand,
                "left_hand": left_hand,
                "right_knee": right_knee,
                "left_knee": left_knee,
                "right_foot": right_foot,
                "left_foot": left_foot,
                "visibility": visibility_mean,
                "presence": presence_mean,
            }
        )

    df = pd.DataFrame(rows, columns=["right_hand", "left_hand", "right_knee", "left_knee","right_foot", "left_foot", "visibility", "presence"])

    metrics_list = compute_metrics(df, metrics)

    metrics.update(metrics_list)

    return metrics

    
