#!/usr/bin/env bash

# Activate conda env and run crosswalk filtered demo on GPU 7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate roadradar

CUDA_VISIBLE_DEVICES=7 python mytools/demo_track_crosswalk.py image \
  -f exps/example/mot/yolox_x_mix_det.py \
  -c pretrained/bytetrack_x_mot17.pth.tar \
  --path data/day/2025-12-02-16-37-20     \
  --json data/test/zebra_points.json \
  --draw_border \
  --save_result \
  --fp16  \
  --fuse
