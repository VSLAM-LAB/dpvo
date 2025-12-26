import os
import cv2
import yaml
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
from pathlib import Path
from itertools import chain

def load_calibration(calibration_yaml: Path, cam_name: str):
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;
    print(f"\nCamera Name: {cam['cam_name']}")
    print(f"Camera Type: {cam['cam_type']}")
    print(f"Camera Model: {cam['cam_model']}")
    print(f"Focal Length: {cam['focal_length']}")
    print(f"Principal Point: {cam['principal_point']}")
    has_dist = ('distortion_type' in cam) and ('distortion_coefficients' in cam)
    if has_dist:
        print(f"Distortion Type Dimension: {cam['distortion_type']}")
        print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
    print(f"Image Dimension: {cam['image_dimension']}")
    print(f"Fps: {cam['fps']}")

    K = np.array([[cam['focal_length'][0], 0,  cam['principal_point'][0]],
                  [0,  cam['focal_length'][1], cam['principal_point'][1]],
                  [0,  0,   1]], dtype=np.float32)
    
    if has_dist:
        dist = np.array(cam['distortion_coefficients'], dtype=np.float32)
    else:
        dist = 0.0

    return K, dist, cam['image_dimension'][0], cam['image_dimension'][1]


def image_stream(queue, sequence_path, rgb_csv, calibration_yaml, cam_name = "rgb0", target_pixels: int = 384*512):
    """ image generator """

    K, dist, w0, h0 = load_calibration(calibration_yaml = calibration_yaml, cam_name = cam_name)

    # Load rgb images
    df = pd.read_csv(rgb_csv)       
    
    image_list = df[f'path_{cam_name}'].to_list()
    timestamps = (df[f'ts_{cam_name} (ns)'] / 1e9).to_list()

    # Undistort and resize images
    h = (int(h0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32
    w = (int(w0 * np.sqrt(target_pixels / (h0 * w0))) // 32) * 32
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w0, h0), 0, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
    intrinsics = np.array([new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2]])
    
    # Load images
    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(os.path.join(sequence_path, imfile)))
        image = cv2.remap(image, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        queue.put((timestamps[t], image, intrinsics))

    queue.put((-1, image, intrinsics))

def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    assert os.path.exists(imagedir), imagedir
    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

