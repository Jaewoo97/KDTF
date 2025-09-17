import os
from tqdm import tqdm
import shutil

root_dir = "/mnt/jaewoo4tb/textraj/preprocessed_2nd/sit_v2_fps_2_5_frame_20_withRobot_withPoseInfo_v2"
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

# 0: Cafe_street_1-002_agents_0_to_200 / outdoor
# 4: Cafeteria_3-004_agents_0_to_200   / indoor
# 7: Corridor_1-010_agents_0_to_200 / indoor
# 36: Lobby_6-001_agents_0_to_200 / indoor
# 40: Outdoor_Alley_3-002_agents_0_to_200 / outdoor
# 44: Three_way_Intersection_4-001_agents_0_to_200 / outdoor
# 41: Subway_Entrance_2-004 / outdoor

val_idxs = [0, 4, 7, 36, 40, 44, 41]

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