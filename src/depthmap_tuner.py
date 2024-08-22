#!/usr/bin/env python3

import os

import cv2
import numpy as np
from stereovision.calibration import StereoCalibration

from camera import StereoCamera


trackbar_load_status = False


def stereo_depth_map(rectified_pair, variable_mapping):
    sbm = cv2.StereoBM().create(numDisparities=16, blockSize=variable_mapping["SWS"])
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(variable_mapping["PreFiltSize"])
    sbm.setPreFilterCap(variable_mapping["PreFiltCap"])
    sbm.setSpeckleRange(variable_mapping["SpeckleRange"])
    sbm.setSpeckleWindowSize(variable_mapping["SpeckleSize"])
    sbm.setMinDisparity(variable_mapping["MinDisp"])
    sbm.setNumDisparities(variable_mapping["NumofDisp"])
    sbm.setTextureThreshold(variable_mapping["TxtrThrshld"])
    sbm.setUniquenessRatio(variable_mapping["UniqRatio"])

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    disparity_normalized = cv2.normalize(disparity, disparity, 0, 255, cv2.NORM_MINMAX)

    image = np.array(disparity_normalized, dtype=np.uint8)
    disparity_color = cv2.applyColorMap(image, cv2.COLORMAP_JET)

    return disparity_color, disparity_normalized


def on_change_trackbar(x: int) -> None:
    global trackbar_load_status
    trackbar_load_status = False


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

    tune_window_name = "Depthmap tuner"
    cv2.namedWindow(tune_window_name)
    cv2.createTrackbar("SWS", tune_window_name, 115, 230, on_change_trackbar)
    cv2.createTrackbar("SpeckleSize", tune_window_name, 0, 300, on_change_trackbar)
    cv2.createTrackbar("SpeckleRange", tune_window_name, 0, 40, on_change_trackbar)
    cv2.createTrackbar("UniqRatio", tune_window_name, 1, 20, on_change_trackbar)
    cv2.createTrackbar("TxtrThrshld", tune_window_name, 0, 1000, on_change_trackbar)
    cv2.createTrackbar("NumofDisp", tune_window_name, 1, 16, on_change_trackbar)
    cv2.createTrackbar("MinDisp", tune_window_name, -100, 200, on_change_trackbar)
    cv2.createTrackbar("PreFiltCap", tune_window_name, 1, 63, on_change_trackbar)
    cv2.createTrackbar("PreFiltSize", tune_window_name, 5, 255, on_change_trackbar)
    cv2.createTrackbar("Save Settings", tune_window_name, 0, 1, on_change_trackbar)
    cv2.createTrackbar("Load Settings", tune_window_name, 0, 1, on_change_trackbar)

    variables = [
        "SWS",
        "SpeckleSize",
        "SpeckleRange",
        "UniqRatio",
        "TxtrThrshld",
        "NumofDisp",
        "MinDisp",
        "PreFiltCap",
        "PreFiltSize",
    ]

    variable_mapping = {
        "SWS": 15,
        "SpeckleSize": 100,
        "SpeckleRange": 15,
        "UniqRatio": 10,
        "TxtrThrshld": 100,
        "NumofDisp": 1,
        "MinDisp": -25,
        "PreFiltCap": 30,
        "PreFiltSize": 105,
    }

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

            if not trackbar_load_status:
                for v in variables:
                    current_value = cv2.getTrackbarPos(v, tune_window_name)
                    if v == "SWS" or v == "PreFiltSize":
                        if current_value < 5:
                            current_value = 5
                        if current_value % 2 == 0:
                            current_value += 1

                    if v == "NumofDisp":
                        if current_value == 0:
                            current_value = 1
                        current_value = current_value * 16
                    if v == "MinDisp":
                        current_value = current_value - 100
                    if v == "UniqRatio" or v == "PreFiltCap":
                        if current_value == 0:
                            current_value = 1

                    variable_mapping[v] = current_value

            disparity_color, disparity_normalized = stereo_depth_map(
                rectified_pair, variable_mapping
            )

            cv2.imshow(tune_window_name, disparity_color)
            cv2.imshow("Frame", np.hstack((rectified_pair[0], rectified_pair[1])))

            if cv2.waitKey(1) == ord("q"):
                break
            else:
                continue

    camera.stop()
    camera.release()
    cv2.destroyAllWindows()
