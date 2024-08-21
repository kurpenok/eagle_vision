#!/usr/bin/env python3

import os
import time

import cv2
from tqdm import tqdm

from camera import StereoCamera


COUNTDOWN = 3
DELAY = 2


def take_pictures(sensor_id: int, photos_count: int) -> None:
    photos_folder_path = "images"
    if not os.path.isdir(photos_folder_path):
        os.makedirs(photos_folder_path)

    camera = StereoCamera(sensor_id)
    camera.start()

    time.sleep(COUNTDOWN)

    for photo in tqdm(range(photos_count), bar_format="[+] Photos: |{bar:50}|"):
        time.sleep(DELAY)

        grabbed, frame = camera.read()
        if grabbed:
            cv2.imwrite(f"{photos_folder_path}/left_image_{photo}.png", frame[:, :640])
            cv2.imwrite(f"{photos_folder_path}/right_image_{photo}.png", frame[:, 640:])

            cv2.imshow("[+] Camera image:", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    camera.stop()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    if input("[>] Do you want to start taking photos? (y/n) ").lower() == "y":
        sensor_id = int(input("[>] Enter sensor ID: "))
        photos_count = int(input("[>] Enter photos count: "))

        take_pictures(sensor_id, photos_count)
