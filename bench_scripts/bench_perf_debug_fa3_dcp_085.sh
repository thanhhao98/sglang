#!/bin/bash

# Get current branch and the first 5 characters of the commit hash
BRANCH=$(git rev-parse --abbrev-ref HEAD)
HASH=$(git rev-parse --short=7 HEAD)
FOLDER_PATH=bench_result
export OUTPUT_NAME="${FOLDER_PATH}/${BRANCH}_${HASH}/tp8_dcp8_fa3_a2a_vectorized_085/"
mkdir -p $OUTPUT_NAME

# Define the concurrency levels
CONCURRENCIES=(1 2 4 8 16 64 256 512)

for C in "${CONCURRENCIES[@]}"; do
    # Calculate num-prompts (Concurrency * 5)
    NUM_PROMPTS=$((C * 5))
    FILE_NAME="${OUTPUT_NAME}cc${C}.txt"

    echo "-------------------------------------------------------"
    echo "Branch: $BRANCH | Hash: $HASH"
    echo "Running benchmark: Concurrency=$C, Prompts=$NUM_PROMPTS"
    echo "Writing result to $FILE_NAME"
    echo "-------------------------------------------------------"

    python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port 8188 \
        --model deepseek-ai/DeepSeek-V2 \
        --dataset-name random \
        --random-input-len 4000 \
        --random-output-len 1500 \
        --random-range-ratio 0.1 \
        --num-prompts $NUM_PROMPTS \
        --max-concurrency $C \
	--disable-ignore-eos 2>&1 | tee "$FILE_NAME"

done
