"""Stage 3: arrange stage-2 samples into the <root>/train and <root>/val folders read by HiVT/datasets/jrdb_dataset.py.

Files are renamed to `<scene_idx:03d>_<initial_frame:04d>.pt`, where scene_idx is the index of the scene in the
sorted list of scene folders (the same index stored as `scene_idx` inside JRDB/SIT samples).

Examples:
    python preprocessing/split_train_val.py --dataset jrdb --src data/stage2/jrdb --dst data/jrdb_v2_fps_2_5_frame_20_withRobot
    # ETH/UCY leave-one-scene-out: build one root per held-out scene
    for s in eth hotel zara1 zara2 students; do
        python preprocessing/split_train_val.py --dataset ethucy --val_scenes $s --src data/stage2/ethucy --dst data/ethucy_v2_fps_2_5_frame_20_$s
    done
"""
import argparse
import os
import shutil

from tqdm import tqdm

# Validation scenes used in the paper
DEFAULT_VAL_SCENES = {
    'jrdb': ['bytes-cafe-2019-02-07_0', 'huang-lane-2019-02-12_0', 'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_0'],
    'sit': ['Cafe_street_1-002', 'Cafeteria_3-004', 'Corridor_1-010', 'Lobby_6-001', 'Outdoor_Alley_3-002',
            'Subway_Entrance_2-004', 'Three_way_Intersection_4-001'],
    'ethucy': None,  # leave-one-out, must be given with --val_scenes
}
ETHUCY_SCENES = ['eth', 'hotel', 'zara1', 'zara2', 'students']


def scene_key(folder_name):
    """`bytes-cafe-2019-02-07_0_agents_0_to_1726` -> `bytes-cafe-2019-02-07_0`."""
    return folder_name.split('_agents_')[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, choices=list(DEFAULT_VAL_SCENES.keys()))
    parser.add_argument('--src', required=True, help='stage2_build_samples.py output (one folder per scene)')
    parser.add_argument('--dst', required=True, help='Dataset root to create (gets train/ and val/)')
    parser.add_argument('--val_scenes', nargs='+', default=None, help='Override the validation scenes')
    parser.add_argument('--mode', default='symlink', choices=['symlink', 'copy', 'move'],
                        help='symlink (default) keeps a single copy of the data in --src')
    args = parser.parse_args()

    val_scenes = args.val_scenes or DEFAULT_VAL_SCENES[args.dataset]
    if val_scenes is None:
        parser.error(f'--val_scenes is required for ethucy (one of {ETHUCY_SCENES})')

    scenes = sorted(d for d in os.listdir(args.src) if os.path.isdir(os.path.join(args.src, d)) and d not in ('train', 'val'))
    keys = [scene_key(s) for s in scenes]
    unknown = set(val_scenes) - set(keys)
    if unknown:
        raise ValueError(f'Validation scenes not found in {args.src}: {sorted(unknown)}')

    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.dst, split), exist_ok=True)
    counts = {'train': 0, 'val': 0}
    for idx, (scene, key) in enumerate(tqdm(list(zip(scenes, keys)))):
        split = 'val' if key in val_scenes else 'train'
        for file in os.listdir(os.path.join(args.src, scene)):
            if not file.endswith('.pt'):
                continue
            src = os.path.abspath(os.path.join(args.src, scene, file))
            dst = os.path.join(args.dst, split, f'{idx:03d}_{file[:-3].zfill(4)}.pt')
            if args.mode == 'symlink':
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            elif args.mode == 'copy':
                shutil.copy(src, dst)
            else:
                shutil.move(src, dst)
            counts[split] += 1
    print(f'val scenes: {val_scenes}')
    print(f'train: {counts["train"]} samples, val: {counts["val"]} samples -> {args.dst}')


if __name__ == '__main__':
    main()
