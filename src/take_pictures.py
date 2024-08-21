#!/usr/bin/env python3

import os
import time

import cv2
from tqdm import tqdm

from camera import StereoCamera


COUNTDOWN = 4
DELAY = 2


def take_pictures(
    sensor_id: int,
    frame_width: int,
    frame_height: int,
    photos_count: int,
    photos_folder_path: str,
) -> None:
    if not os.path.isdir(photos_folder_path):
        os.makedirs(photos_folder_path)

    camera = StereoCamera(sensor_id, frame_width, frame_height)
    camera.start()

    time.sleep(COUNTDOWN)

    for photo in tqdm(range(photos_count), bar_format="[+] Photos: |{bar:50}|"):
        time.sleep(DELAY)

        capture = camera.read()
        grabbed = capture.grabbed
        frame = capture.frame

        if grabbed:
            left_frame_path = f"{photos_folder_path}/left_image_{photo:02d}.png"
            right_frame_path = f"{photos_folder_path}/right_image_{photo:02d}.png"

            left_frame = frame[:, : frame_width // 2]
            right_frame = frame[:, frame_width // 2 :]

            cv2.imwrite(left_frame_path, left_frame)
            cv2.imwrite(right_frame_path, right_frame)

            cv2.imshow("[+] Camera image:", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    if input("[>] Do you want to start taking photos? (y/n) ").lower() == "y":
        # sensor_id = int(input("[>] Enter sensor ID: "))
        # photos_count = int(input("[>] Enter photos count: "))
        # photos_folder_path = str(input("[>] Enter photos folder path: "))
        # frame_width = int(input("[>] Enter frame width: "))
        # frame_height = int(input("[>] Enter frame height: "))

        sensor_id = 0
        photos_count = 30
        photos_folder_path = "images"
        frame_width = 1280
        frame_height = 480

        take_pictures(
            sensor_id,
            frame_width,
            frame_height,
            photos_count,
            photos_folder_path,
        )
