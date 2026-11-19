import cv2
import numpy as np


def frame_difference(curr_frame: np.ndarray, prev_frame: np.ndarray, thresh: int = 25) -> np.ndarray:
    """Phát hiện thay đổi giữa 2 khung hình liên tiếp (Frame differencing)."""
    g1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    g2 = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if curr_frame.ndim == 3 else curr_frame
    diff = cv2.absdiff(g2, g1)
    _, mask = cv2.threshold(diff, int(thresh), 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


class OpticalFlowTracker:
    """Theo vết chuyển động bằng Lucas-Kanade (goodFeaturesToTrack + calcOpticalFlowPyrLK)."""

    def __init__(self, max_corners: int = 100):
        self.max_corners = max_corners
        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self.trail_mask: np.ndarray | None = None

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None
        self.trail_mask = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=0.05, minDistance=7, blockSize=7
            )
            self.trail_mask = np.zeros_like(vis)
            return vis

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        if next_pts is None or status is None:
            self.reset()
            return vis

        good_new = next_pts[status == 1]
        good_old = self.prev_pts[status == 1]

        for (new_pt, old_pt) in zip(good_new, good_old):
            a, b = int(new_pt[0]), int(new_pt[1])
            c, d = int(old_pt[0]), int(old_pt[1])
            cv2.line(self.trail_mask, (a, b), (c, d), (0, 255, 0), 2)
            cv2.circle(vis, (a, b), 3, (0, 0, 255), -1)

        out = cv2.add(vis, self.trail_mask)
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)
        return out
