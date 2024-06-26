#!/bin/bash
echo "in bash script"
export PATH=$PATH:/opt/conda/envs/urlb2/bin:/opt/conda/bin
echo "all arguments: $@"
eval "$(conda shell.bash hook)"
conda activate urlb2

if [ $1 = "fine" ]; then
	python train_offline.py ${@:2}
else 
	python pretrain_dreamer.py ${@:2}
fi

