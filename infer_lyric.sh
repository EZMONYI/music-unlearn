#!/bin/bash

data_dir=${1:-"lmd_data/processed"}
user_dir=${2:-"mass"}
model=${3:-"checkpoints/checkpoint_best.pt"}

fairseq-generate $data_dir \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-langs melody --target-langs lyric \
  --langs lyric,melody \
  --source-lang melody --target-lang lyric \
  --mt_steps melody-lyric \
  --gen-subset train \
  --beam 5 \
  --nbest 5 \
  --remove-bpe \
  --max-len-b 500 \
  --no-early-stop \
  --path $model \
  --sampling