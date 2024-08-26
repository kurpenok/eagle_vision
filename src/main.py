#!/usr/bin/env python3

import os

import cv2
import numpy as np
from stereovision.calibration import StereoCalibration

from camera import StereoCamera

SWS = 29
PFS = 5
PFC = 20
MDS = 0
NOD = 80
TTH = 500
UR = 1
SR = 0
SPWS = 0


def load_map_settings():
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings, sbm
    sbm = cv2.StereoBM().create(numDisparities=16, blockSize=SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)


def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)
    image = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return disparity_color, disparity_normalized


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    # sensor_id = int(input("[>] Enter sensor ID: "))
    # frame_width = int(input("[>] Enter frame width: "))
    # frame_height = int(input("[>] Enter frame height: "))

    sensor_id = 2
    frame_width = 1280
    frame_height = 480

    camera = StereoCamera(sensor_id, frame_width, frame_height)
    camera.start()
    load_map_settings()

    cv2.namedWindow("Depthmap")

    while True:
        capture = camera.read()
        grabbed = capture.grabbed
        frame = capture.frame

        if grabbed:
            left_frame = frame[:, : frame_width // 2]
            right_frame = frame[:, frame_width // 2 :]

            left_gray_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            calibration_result_folder_path = "calibration_result"
            calibration = StereoCalibration(input_folder=calibration_result_folder_path)
            rectified_pair = calibration.rectify((left_gray_frame, right_gray_frame))
            disparity_color, disparity_normalized = stereo_depth_map(rectified_pair)

            output = cv2.addWeighted(left_frame, 0.5, disparity_color, 0.5, 0.0)
            cv2.imshow("DepthMap", np.hstack((disparity_color, output)))

            if cv2.waitKey(1) == ord("q"):
                break
            else:
                continue

    camera.stop()
    camera.release()
    cv2.destroyAllWindows()
