#!/bin/bash

model=${1:-"../checkpoints/checkpoint_best.pt"}
split=${2:-"train"}  # train = unlearn

python ../fairseq-0.10.2/fairseq_cli/generate.py "../data/lmd_data/processed" \
  --user-dir "../mass" \
  --task xmasked_seq2seq \
  --source-langs melody --target-langs lyric \
  --langs lyric,melody \
  --source-lang melody --target-lang lyric \
  --mt_steps melody-lyric \
  --gen-subset $split \
  --beam 5 \
  --nbest 5 \
  --remove-bpe \
  --max-len-b 500 \
  --no-early-stop \
  --path $model \
  --sampling
