import os
from tqdm import tqdm
import shutil

root_dir = "/mnt/jaewoo4tb/textraj/preprocessed_2nd/jrdb_v2_fps_2_5_frame_20_egovlm"
train_dir = f"{root_dir}/train"
val_dir = f"{root_dir}/val"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

scenes = os.listdir(root_dir)
scenes.sort()
scenes.remove("train")
scenes.remove("val")

val_idxs = [0, 13, 14, 18] # bytes cafe, huang lane, jordan hall, poster session 0 => v2_fps_2_5_frame_20

for idx, scene in tqdm(enumerate(scenes)):
    sub_dir = f"{root_dir}/{scene}"
    files = os.listdir(sub_dir)
    for file in files:
        if file[-3:] != ".pt":
            continue

        src = f"{sub_dir}/{file}"
        if idx in val_idxs:
            dst = f"{val_dir}/{str(idx).zfill(3)}_{file[:-3].zfill(4)}.pt"
        else:
            dst = f"{train_dir}/{str(idx).zfill(3)}_{file[:-3].zfill(4)}.pt"

        # shutil.copy(src, dst)
        shutil.move(src, dst)