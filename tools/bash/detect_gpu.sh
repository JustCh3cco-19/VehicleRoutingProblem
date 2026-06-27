#!/bin/bash
# Detect local GPU and output the corresponding CUDA SM architecture flag

if ! command -v nvidia-smi &> /dev/null; then
	echo "sm_75"
	exit 0
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

if [[ "$GPU_NAME" == *"1650"* ]]; then
	echo "sm_75"
elif [[ "$GPU_NAME" == *"6000"* || "$GPU_NAME" == *"Quadro"* || "$GPU_NAME" == *"A6000"* ]]; then
	if [[ "$GPU_NAME" == *"Ada"* ]]; then
		echo "sm_89"
	else
		echo "sm_86"
	fi
elif [[ "$GPU_NAME" == *"2070"* ]]; then
	echo "sm_75"
else
	echo "sm_75"
fi
