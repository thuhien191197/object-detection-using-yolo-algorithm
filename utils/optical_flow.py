import cv2
import numpy as np

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS |
                           cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class Optical:
    def __init__(self):
        self.p0 = np.array([[[0, 0]]], dtype=np.float32)

    def update(self, old_frame, new_frame):
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_frame, new_frame, self.p0, None, **lk_params)
        return p1
