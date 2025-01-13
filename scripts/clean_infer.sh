#!/bin/bash

ckpt=${1:-"../checkpoints/checkpoint_best.pt"}
split=${2:-"train"}

sh infer_lyric.sh $ckpt $split | grep ^H | cut -f1,2,3 > ../results/lyric.inf
sh infer_lyric.sh $ckpt $split | grep ^T | cut -f1,2 > ../results/lyric.ref
sh infer_melody.sh $ckpt $split | grep ^H | cut -f1,2,3 > ../results/melody.inf
sh infer_melody.sh $ckpt $split | grep ^T | cut -f1,2 > ../results/melody.ref
