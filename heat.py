from utils import read_video, save_video
from trackers import Tracker
from team_assigners import TeamAssigner
import cv2
import numpy as np




    # 4 points from your video frame (pixel coordinates)
pts_src = np.array([[100, 200],
                        [500, 200],
                        [500, 400],
                        [100, 400]], dtype='float32')

    # 4 corresponding points on the pitch (e.g. in meters)
pts_dst = np.array([[0, 0],
                        [100, 0],
                        [100, 50],
                        [0, 50]], dtype='float32')

    # Compute homography matrix
H, status = cv2.findHomography(pts_src, pts_dst)
print(H)
