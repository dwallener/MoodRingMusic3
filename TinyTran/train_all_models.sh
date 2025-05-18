#!/bin/bash

# cello, violo, violin1, violin2 (melody)

TRAIN_SCRIPT="train_phrase_transformer.py"
PHRASE_TYPES=("short" "medium" "long")

for TYPE in "${PHRASE_TYPES[@]}"; do
    DATASET="cello_phrase_dataset_${TYPE}.json"
    MODEL_OUTPUT="tiny_transformer_cello_${TYPE}.pt"
    VOCAB_OUTPUT="tiny_transformer_cello_${TYPE}.vocab.json"

    # Capitalize first letter manually for display
    if [ "$TYPE" = "short" ]; then
        TYPE_CAP="Short"
    elif [ "$TYPE" = "medium" ]; then
        TYPE_CAP="Medium"
    else
        TYPE_CAP="Long"
    fi

    echo "Training Cello ${TYPE_CAP} Model..."
    python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT
done

echo "All cello models trained successfully."
