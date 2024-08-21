#!/usr/bin/env python3

import os

import cv2
from tqdm import tqdm
from stereovision.calibration import StereoCalibrator
from stereovision.calibration import StereoCalibration
from stereovision.exceptions import ChessboardNotFoundError


def calibration(
    photos_folder_path: str,
    photos_count: int,
    rows: int,
    columns: int,
    square_size: float,
    image_width: int,
    image_height: int,
) -> None:
    if not os.path.isdir(photos_folder_path):
        print("[-] Specified folder does not exist!")
        return

    calibrator = StereoCalibrator(
        rows,
        columns,
        square_size,
        (image_width, image_height),
    )

    left_image = None
    right_image = None
    for photo in tqdm(range(photos_count), bar_format="[+] Photos: |{bar:50}|"):
        left_image_name = f"{photos_folder_path}/left_image_{photo:02d}.png"
        right_image_name = f"{photos_folder_path}/right_image_{photo:02d}.png"

        if os.path.isfile(left_image_name) and os.path.isfile(right_image_name):
            left_image = cv2.imread(left_image_name, 1)
            right_image = cv2.imread(right_image_name, 1)

            try:
                calibrator._get_corners(left_image)
                calibrator._get_corners(right_image)
            except ChessboardNotFoundError:
                print(f"[-] {photo} pair skipped!")
            else:
                calibrator.add_corners((left_image, right_image), True)
        else:
            print("[-] Photos pair not found!")

    if left_image is not None and right_image is not None:
        print("[+] Starting calibration! Wait...")
        calibration = calibrator.calibrate_cameras()
        calibration_result_folder_path = "calibration_result"
        calibration.export(calibration_result_folder_path)
        print("[+] Calibration complete!")

        calibration = StereoCalibration(input_folder=calibration_result_folder_path)
        rectified_pair = calibration.rectify((left_image, right_image))

        cv2.imshow("[+] Left image calibrated!", rectified_pair[0])
        cv2.imshow("[+] Right image calibrated!", rectified_pair[1])

        cv2.imwrite(
            f"{calibration_result_folder_path}/calibrated_left_image.png",
            rectified_pair[0],
        )
        cv2.imwrite(
            f"{calibration_result_folder_path}/calibrated_right_image.png",
            rectified_pair[1],
        )

        cv2.waitKey(0)
    else:
        print("[-] Calibration error!")


if __name__ == "__main__":
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    # photos_folder_path = str(input("[>] Enter photos folder path: "))
    # total_photos = int(input("[>] Enter count of photos pairs: "))
    # rows = int(input("[>] Enter intersections by row count: "))
    # columns = int(input("[>] Enter intersections by column count: "))
    # square_size = int(input("[>] Enter square size: "))

    photos_folder_path = "images"
    photos_count = 30
    rows = 7
    columns = 10
    square_size = 2
    image_width = 640
    image_height = 480

    calibration(
        photos_folder_path,
        photos_count,
        rows,
        columns,
        square_size,
        image_width,
        image_height,
    )
