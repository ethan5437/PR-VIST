#!/bin/bash
set -x
CUDA_LAUNCH_BLOCKING=1 python train.py -log $2 -save_model trained -save_mode all -label_smoothing -device $3 -model $1 -vist -hop 1.5 -loss_level hierarchical -reward_rate 1 -is_story_discriminator 
