import os
from multiprocessing import Process, Queue
from pathlib import Path
import csv
import yaml

import cv2
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, sequence_path, rgb_csv, calibration_yaml, viz=False, timeit=False, cam_name="rgb_0"):

    slam = None
    queue = Queue(maxsize=8)

    reader = Process(target=image_stream, args=(queue, sequence_path, rgb_csv, calibration_yaml, cam_name))
    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            slam(t, image, intrinsics)

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]
    
    return slam.terminate(), (points, colors, (*intrinsics, H, W))

def main():

    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")

    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--loop_closure', type=int, default=1, choices=[0, 1, 2],
                        help='0: DPVO (odometry only), 1: DPV-SLAM (proximity loop closure), '
                             '2: DPV-SLAM++ (proximity + classic DBoW2 loop closure, needs --orb_vocab)')
    parser.add_argument('--orb_vocab', type=Path, default=None, help='ORBvoc.txt for --loop_closure 2')

    args, _ = parser.parse_known_args()

    cfg.merge_from_file(args.settings_yaml)
    cfg.LOOP_CLOSURE = args.loop_closure >= 1
    cfg.CLASSIC_LOOP_CLOSURE = args.loop_closure == 2
    if args.orb_vocab is not None:
        cfg.ORB_VOCAB = str(args.orb_vocab)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(args.settings_yaml)
    cam_name = cfg.CAM_MONO
 
    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, 
                                                    args.sequence_path, args.rgb_csv, args.calibration_yaml, 
                                                    bool(int(args.verbose)), args.timeit, cam_name = cam_name)

    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        for i in range(len(tstamps)):
            ts = int(tstamps[i] * 1e9)
            tx, ty, tz, qx, qy, qz, qw = poses[i]
            writer.writerow([ts, tx, ty, tz, qx, qy, qz, qw])

if __name__ == '__main__':
    main()

        

