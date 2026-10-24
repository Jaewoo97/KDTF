# KDTF

Knowledge distillation for human trajectory forecasting. A **teacher** forecaster sees each agent's past
trajectory, its 3D body pose and a natural-language description of its behaviour. It is then distilled into a
**student** that only needs the trajectory (and optionally the pose) at test time. Both are built on
[HiVT](https://github.com/ZikangZhou/HiVT).

Supported datasets: **JRDB**, **SIT** and **ETH/UCY** (leave-one-scene-out).

```
raw dataset ──stage 1──▶ per-scene dict (positions, pose, captions)
            ──stage 2──▶ HiVT TemporalData samples (.pt, one per 20-step window)
            ──split────▶ <root>/train, <root>/val
            ──train_teacher.py──▶ teacher.ckpt ──train_kd.py──▶ student.ckpt
```

## Repository layout

```
HiVT/
  train_teacher.py      # stage A: multi-modal teacher
  train_kd.py           # stage B: distill the frozen teacher into the student
  validate.py           # evaluate a teacher / KD checkpoint on <root>/val
  models/               # HiVT variants, encoders and KD models
  datasets/ datamodules/ losses/ metrics/
  _train_*.py           # baselines (MART, LED, Social-Transmotion); not cleaned up
preprocessing/
  stage1_jrdb.py  stage1_sit.py  stage1_ethucy.py  ethucy_extract_frames.py
  stage2_build_samples.py
  split_train_val.py
  jrdb/odometry_processed/   # robot odometry per JRDB scene (used by stage1_jrdb.py)
parser_model/  bev/  romp/  utils/   # captioning / BEV pose engines used in stage 1
kdtf_paths.py           # locations of external assets (overridable by env vars)
```

## Installation

Tested with Python 3.11, PyTorch 2.3.0 (CUDA 12.1), PyG 2.5.3, PyTorch Lightning 1.9.0 and transformers 4.26.1.
Training requires a GPU.

```bash
conda create -n kdtf python=3.11 -y && conda activate kdtf
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

PyTorch Lightning must stay **< 2.0**: the scripts use `Trainer.from_argparse_args` and `--gpus`.

Stage 1 needs a few extra assets. Stage 2 and training do not.

| asset | used by | where |
|---|---|---|
| BEV / SMPL-A files: `BEV.pth`, `SMPLA_NEUTRAL.pth`, `smil_packed_info.pth` (optionally `ROMP.pkl`, `SMPL_NEUTRAL.pth`) | JRDB, SIT (3D pose) | [ROMP releases](https://github.com/Arthur151/ROMP/releases) / [simple_romp instructions](https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation); put them in `romp/files/.romp/` or set `KDTF_ROMP_DIR` |
| PLLaVA-13B weights | SIT (video captions) | [PLLaVA](https://github.com/magic-research/PLLaVA); set `PLLAVA_MODEL_DIR`. Run SIT stage 1 in an environment that satisfies PLLaVA's own requirements |
| `openai/clip-vit-base-patch16`, `huawei-noah/TinyBERT_General_4L_312D` | ETH/UCY stage 1, stage 2 | downloaded automatically from the Hugging Face Hub |

`KDTF_TMP_DIR` (default `./temp`) holds temporary image crops.

## Data preparation

All commands are run from the repository root. Stage 2 encodes every caption with TinyBERT (312-d [CLS]).

### JRDB

Expected raw layout (`--jrdb_root`):

```
train_dataset/
  images/image_0/<scene>/            # only used to list the scenes
  images/image_stitched/<scene>/*.jpg
  labels/labels_2d_stitched/<scene>.json
  labels/labels_3d/<scene>.json
  labels/labels_2d_activity_social_stitched/<scene>.json   # JRDB-Act / JRDB-Social labels (JRDB 2024 labels.zip)
```

Robot odometry for every training scene is included in `preprocessing/jrdb/odometry_processed/`.
Agent captions and pairwise-interaction captions are built from the JRDB-Act/Social labels; no VLM is needed.

```bash
python preprocessing/stage1_jrdb.py --jrdb_root /path/to/jrdb/train_dataset --output_dir data/preprocessed_1st/jrdb_v1
python preprocessing/stage2_build_samples.py --dataset jrdb --input_dir data/preprocessed_1st/jrdb_v1 --output_dir data/stage2/jrdb
python preprocessing/split_train_val.py --dataset jrdb --src data/stage2/jrdb --dst data/jrdb_v2_fps_2_5_frame_20_withRobot
```

Validation scenes: `bytes-cafe-2019-02-07_0`, `huang-lane-2019-02-12_0`, `jordan-hall-2019-04-22_0`,
`packard-poster-session-2019-03-20_0`. All other scenes are used for training.

### SIT

Expected raw layout (`--sit_root`): one folder per scene containing `cam_img/<cam>/data_rgb/*.png`,
`ego_trajectory/`, `label_2d/` and `label_3d/`. Scenes without all four are skipped.

```bash
# PLLaVA captioning is slow; run one scene per job (indices follow sorted scene names)
for i in $(seq 0 62); do
  python preprocessing/stage1_sit.py --sit_root /path/to/sit --output_dir data/preprocessed_1st/sit_v1_prompt3 --scene_idx $i
done
python preprocessing/stage2_build_samples.py --dataset sit --input_dir data/preprocessed_1st/sit_v1_prompt3 --output_dir data/stage2/sit
python preprocessing/split_train_val.py --dataset sit --src data/stage2/sit --dst data/sit_v2_fps_2_5_frame_20_withRobot
```

Validation scenes: `Cafe_street_1-002`, `Cafeteria_3-004`, `Corridor_1-010`, `Lobby_6-001`,
`Outdoor_Alley_3-002`, `Subway_Entrance_2-004`, `Three_way_Intersection_4-001`.

### ETH/UCY

Expected raw layout (`--ethucy_root`):

```
biwi_eth.avi  biwi_hotel.avi  crowds_zara01.avi  crowds_zara02.avi  students003.avi
biwi_eth.txt  biwi_hotel.txt  crowds_zara01.txt  crowds_zara02.txt  students003.txt   # world-coordinate tracks (format as in SHENet)
ynet/{eth,hotel,zara1,zara2,students}_oracle.png                                     # Y-Net semantic maps
```

Captions are rule-based: motion state, plus the direction of the closest obstacle on the semantic map. The
"pose" input is the CLIP ViT-B/16 feature of the agent's image crop (512-d).

```bash
python preprocessing/ethucy_extract_frames.py --ethucy_root /path/to/ethucy          # -> /path/to/ethucy/images/<scene>/
python preprocessing/stage1_ethucy.py --ethucy_root /path/to/ethucy --output_dir data/preprocessed_1st/ethucy_v1
python preprocessing/stage2_build_samples.py --dataset ethucy --input_dir data/preprocessed_1st/ethucy_v1 --output_dir data/stage2/ethucy
# leave-one-scene-out: one dataset root per held-out scene
for s in eth hotel zara1 zara2 students; do
  python preprocessing/split_train_val.py --dataset ethucy --val_scenes $s --src data/stage2/ethucy --dst data/ethucy_v2_fps_2_5_frame_20_$s
done
```

### Notes

- Each sample is a 20-step window at 2.5 FPS (8 observed + 12 future steps). JRDB/SIT samples also store the
  robot position, scene index and agent ids.
- `split_train_val.py` creates symlinks by default (`--mode copy|move` are also available). Keep `data/stage2/`
  in place, or use `--mode copy`.
- `stage2_build_samples.py --scene_idx_start/--scene_idx_end` splits the work across processes, and
  `--max_samples_per_scene` builds a small debug set.
- JRDB pairwise-interaction captions are written to the edge features (`x_interaction`) only with
  `--fix_pairwise_interaction`. By default the edges are left masked, which matches the data the reported models
  were trained on.

## Training

Logs and checkpoints go to `--log_dir` (default `./logs`) under `--log_name` (default `teacher_<dataset>` /
`kd_<dataset>`). The best `--save_top_k` checkpoints by `val_minADE` are kept.

**1. Teacher** (trajectory + pose + text)

```bash
cd HiVT
python train_teacher.py --dataset jrdb   --root ../data/jrdb_v2_fps_2_5_frame_20_withRobot
python train_teacher.py --dataset sit    --root ../data/sit_v2_fps_2_5_frame_20_withRobot
python train_teacher.py --dataset ethucy --root ../data/ethucy_v2_fps_2_5_frame_20_eth
```

**2. Knowledge distillation** into the student

```bash
python train_kd.py --dataset jrdb --root ../data/jrdb_v2_fps_2_5_frame_20_withRobot \
    --teacher_ckpt logs/teacher_jrdb/lightning_logs/version_0/checkpoints/<best>.ckpt
# trajectory-only student
python train_kd.py --dataset jrdb --root ... --teacher_ckpt ... --encoder_type_student traj_v3 --log_name kd_jrdb_traj
```

The teacher is frozen. The KD model loads the teacher weights and builds its student and dummy branches from
scratch. All model hyper-parameters (`num_modes`, `embed_dim`, `lr`, ...) are read from the teacher checkpoint.
From the command line, `train_kd.py` only takes `--encoder_type_student` and the data/trainer options
(`--root`, batch sizes, `--max_epochs`, `--gpus`, logging).

Defaults per dataset:

| dataset | teacher model / encoder | KD model | default student | modes | seeds (teacher / KD) |
|---|---|---|---|---|---|
| jrdb | `models/hivt_v3.py` / `traj_pose_text_intra` | `hivt_kd_onlyAgent_v4_dummy_teachV3` | `traj_pose` | 1 | 1110 / 5115 |
| sit | `models/hivt_v3_sit.py` / `traj_pose_text` | `hivt_kd_onlyAgent_sit_v4_dummy` | `traj_pose` | 6 | 1555 / 9657 |
| ethucy | `models/hivt_v2_ethucy.py` / `traj_pose_text` | `hivt_kd_onlyAgent_ethucy_v4_nodummy` | `traj_v3` | 6 | 9999 / 4631 |

Common teacher options: `--max_epochs 64 --train_batch_size 64 --lr 5e-4 --num_modes N --gpus 1`. See
`python train_teacher.py --dataset jrdb -h`.

## Evaluation

```bash
cd HiVT
python validate.py --dataset jrdb --model kd      --root ../data/jrdb_v2_fps_2_5_frame_20_withRobot --ckpt logs/kd_jrdb/.../<ckpt>.ckpt
python validate.py --dataset jrdb --model teacher --root ../data/jrdb_v2_fps_2_5_frame_20_withRobot --ckpt logs/teacher_jrdb/.../<ckpt>.ckpt
```

Reported metrics are `val_minADE` / `val_minFDE`, in meters.

Pretrained checkpoints are not included in this repository.

## Acknowledgements

This code builds on [HiVT](https://github.com/ZikangZhou/HiVT), [ROMP/BEV](https://github.com/Arthur151/ROMP),
[PLLaVA](https://github.com/magic-research/PLLaVA), [TinyBERT](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
and [Y-Net](https://github.com/HarshayuGirase/Human-Path-Prediction).
