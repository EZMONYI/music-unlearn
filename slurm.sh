#!/bin/bash
#SBATCH --job-name=unlearn			 # Job name
#SBATCH --output=logs/%j.txt   # Standard output and error log (%j = job ID)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --mem=20G                    # Memory per node
#SBATCH --partition=audio_mt_asr     # Partition for GPU jobs (adjust as needed)
#SBATCH --gres=gpu:1

# Load necessary modules (if any)
module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearn 

# Run your bash script
data_dir=lmd_data/processed # The path of binarized data
user_dir=mass
tensor_log=runs

fairseq-train $data_dir \
  --tensorboard-logdir runs \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-langs lyric,melody \
  --target-langs lyric,melody \
  --langs lyric,melody \
  --arch xtransformer \
  --mass_steps lyric-lyric,melody-melody \
  --mt_steps lyric-melody,melody-lyric \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --lr 0.00005 --min-lr 1e-09 --warmup-init-lr 1e-07 \
  --criterion label_smoothed_cross_entropy_with_align \
  --attn-loss-weight 1.0 \
  --max-tokens 4096 \
  --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
  --max-epoch 20 \
  --max-update 2000000 \
  --share-decoder-input-output-embed \
  --valid-lang-pairs lyric-lyric,melody-melody \
  --no-epoch-checkpoints \
  --skip-invalid-size-inputs-valid-test
