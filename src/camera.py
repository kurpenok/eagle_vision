#!/usr/bin/env python3

import os
import threading
from typing import Tuple

import cv2
import numpy as np


class Capture:
    def __init__(self, grabbed: bool, frame: np.ndarray) -> None:
        self.grabbed: bool = grabbed
        self.frame: np.ndarray = frame.copy()


class StereoCamera:
    def __init__(self, sensor_id: int) -> None:
        self.sensor_id: int = sensor_id

        self.video_capture: cv2.VideoCapture = cv2.VideoCapture(self.sensor_id)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        grabbed, frame = self.video_capture.read()
        self.capture = Capture(grabbed, frame)

        self.read_thread: threading.Thread
        self.read_lock: threading.Lock = threading.Lock()

        self.running: bool = False

    def start(self) -> None:
        if self.running:
            print("[+] Video capturing already running!")
            return

        if self.video_capture:
            self.running = True
            self.read_thread = threading.Thread(target=self.update_camera, daemon=True)
            self.read_thread.start()

    def stop(self) -> None:
        self.running = False
        self.read_thread.join()

    def update_camera(self) -> None:
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.capture.grabbed = grabbed
                    self.capture.frame = frame
            except RuntimeError:
                print("[-] Could not read image from camera!")

    def read(self) -> Tuple[bool, np.ndarray]:
        with self.read_lock:
            grabbed = self.capture.grabbed
            frame = self.capture.frame.copy()
        return grabbed, frame

    def release(self) -> None:
        if self.video_capture:
            self.video_capture.release()

        if self.read_thread:
            self.read_thread.join()


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    sensor_id = int(input("[>] Enter sensor ID: "))

    s = StereoCamera(sensor_id)
    s.start()

    while True:
        grabbed, frame = s.read()

        cv2.imshow("[+] Camera image:", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    s.stop()
    s.release()
    cv2.destroyAllWindows()
