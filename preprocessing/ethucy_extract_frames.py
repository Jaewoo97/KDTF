"""Stage 0 (ETH/UCY): dump every 10th video frame (2.5 FPS, matching the annotation rate) as <scene>/<frame:04d>.jpg.

Example:
    python preprocessing/ethucy_extract_frames.py --ethucy_root /data/ethucy
"""
import argparse
import os

import cv2

VIDEOS = {'students': 'students003.avi', 'zara1': 'crowds_zara01.avi', 'zara2': 'crowds_zara02.avi',
          'eth': 'biwi_eth.avi', 'hotel': 'biwi_hotel.avi'}


def video2frames(video_path, out_dir, every=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    count = 0
    while cap.isOpened() and count <= video_length:
        ret, frame = cap.read()
        if not ret:
            continue
        if count % every == 0:
            cv2.imwrite(os.path.join(out_dir, f'{count:04d}.jpg'), frame)
        count += 1
    cap.release()
    print(f'{video_path}: {count} frames read')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ethucy_root', required=True, help='Folder with the five .avi videos')
    parser.add_argument('--image_dir', default=None, help='Default: <ethucy_root>/images')
    args = parser.parse_args()
    image_dir = args.image_dir or os.path.join(args.ethucy_root, 'images')
    for scene, video in VIDEOS.items():
        video2frames(os.path.join(args.ethucy_root, video), os.path.join(image_dir, scene))


if __name__ == '__main__':
    main()
