#!/bin/bash

# cello, violo, violin1, violin2 (melody)

TRAIN_SCRIPT="train_phrase_transformer.py"

INSTRUMENT="cello"

DATASET="${INSTRUMENT}_phrase_dataset_short.json"
MODEL_OUTPUT="Cello_Short.model.pt"
VOCAB_OUTPUT="Cello_Short.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_medium.json"
MODEL_OUTPUT="Cello_Medium.model.pt"
VOCAB_OUTPUT="Cello_Medium.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_long.json"
MODEL_OUTPUT="Cello_Long.model.pt"
VOCAB_OUTPUT="Cello_Long.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

echo "All ${INSTRUMENT} models trained successfully."

INSTRUMENT="viola"

DATASET="${INSTRUMENT}_phrase_dataset_short.json"
MODEL_OUTPUT="Viola_Short.model.pt"
VOCAB_OUTPUT="Viola_Short.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_medium.json"
MODEL_OUTPUT="Viola_Medium.model.pt"
VOCAB_OUTPUT="Viola_Medium.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_long.json"
MODEL_OUTPUT="Viola_Long.model.pt"
VOCAB_OUTPUT="Viola_Long.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

echo "All ${INSTRUMENT} models trained successfully."

INSTRUMENT="violin1"

DATASET="${INSTRUMENT}_phrase_dataset_short.json"
MODEL_OUTPUT="Violin1_Short.model.pt"
VOCAB_OUTPUT="Violin1_Short.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_medium.json"
MODEL_OUTPUT="Violin1_Medium.model.pt"
VOCAB_OUTPUT="Violin1_Medium.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_long.json"
MODEL_OUTPUT="Violin1_Long.model.pt"
VOCAB_OUTPUT="Violin1_Long.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

echo "All ${INSTRUMENT} models trained successfully."

INSTRUMENT="violin2"

DATASET="${INSTRUMENT}_phrase_dataset_short.json"
MODEL_OUTPUT="Violin2_Short.model.pt"
VOCAB_OUTPUT="Violin2_Short.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_medium.json"
MODEL_OUTPUT="Violin2_Medium.model.pt"
VOCAB_OUTPUT="Violin2_Medium.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

DATASET="${INSTRUMENT}_phrase_dataset_long.json"
MODEL_OUTPUT="Violin2_Long.model.pt"
VOCAB_OUTPUT="Violin2_Long.vocab.json"
python3 $TRAIN_SCRIPT $DATASET --model_out $MODEL_OUTPUT --vocab_out $VOCAB_OUTPUT

echo "All ${INSTRUMENT} models trained successfully."

