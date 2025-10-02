#!/bin/bash

screen -ls | grep -o '[0-9]*\.gshub-datasets-caption-[0-9]*' | xargs -I{} screen -S {} -X quit

unset CUDA_VISIBLE_DEVICES

# screen -dmS "gshub-datasets-caption-0" zsh -ic "conda activate zyh; source hfenv; time python src/ltxv_trainer/custom/inference.py --rank 0 --devices 0 0 >> .caption.rank.0.log; exec zsh";
# screen -dmS "gshub-datasets-caption-1" zsh -ic "conda activate zyh; source hfenv; time python src/ltxv_trainer/custom/inference.py --rank 1 --devices 0 0 >> .caption.rank.1.log; exec zsh";

screen -dmS "gshub-datasets-caption-0" zsh -ic "conda activate zyh; source hfenv; time python src/ltxv_trainer/custom/inference.py --rank 0 --devices 0 >> .caption.rank.0.log; exec zsh";
