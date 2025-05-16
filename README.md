# MoodRingMusic3

📚 Project Structure
TinyTran/
├── extract_phrases_for_training.py
├── train_phrase_transformer.py
├── generate_phrase.py
├── orchestration_composition_music21.py
├── orchestration_composition_mido.py
├── phrase_dataset_short.json
├── phrase_dataset_medium.json
├── phrase_dataset_long.json
├── tiny_transformer_phrase_dataset_short.pt
├── tiny_transformer_phrase_dataset_medium.pt
├── tiny_transformer_phrase_dataset_long.pt
└── ... (vocab files)

🚀 Usage

1. Prepare Datasets

Extract phrases from .krn files and split them into datasets:

python extract_phrases_for_training.py /path/to/krn/folder

2. Train Models

Train models for each phrase type (Short, Medium, Long):

python train_phrase_transformer.py phrase_dataset_short.json
python train_phrase_transformer.py phrase_dataset_medium.json
python train_phrase_transformer.py phrase_dataset_long.json

3. Generate Phrases

Generate a phrase from a trained model:

python generate_phrase.py <model_file.pt> <vocab_file.json> <phrase_length> [temperature]
python generate_phrase.py tiny_transformer_phrase_dataset_short.pt tiny_transformer_phrase_dataset_short.vocab.json 16 1.2

4. Orchestrate a Full Composition

Using Music21:

python orchestration_composition_music21.py

Using Mido (preferred for playback reliability):

python orchestration_composition_mido.py

🎛️ Parameters
	•	phrase_length: Controls generated phrase length directly (used during generation).
	•	temperature: Adjusts randomness. Lower for conservative, higher for exploratory output (default: 1.0).

⸻

📖 Notes
	•	All data is normalized to key of C for simplicity.
	•	The system currently focuses on the melodic line. Bass and harmony generation are planned for future versions.
	•	Generated .mid files can be converted to .wav using fluidsynth or played directly in Logic Pro, Ableton, etc.


# ----- everything below here is the archeological record


Fresh start incorporating learnings from MRM and MRM2

The algorithmic portion is locked and loaded. All the features you'd expect - diurnal, mood, user genre preference - are reflected in the generation algorithm.

Maestro v3 is being used to train the TinyTran for classical generation. Three scripts:

- Generate supplementary annotations based on analysis of the compositions
- Complete training and validation loop
- Generator Class to hook into the main generation code

Training script can be run something like this...
```
python composer_melody_trainer.py \
    --meta_csv /path/to/maestro_midi_metadata.csv \
    --midi_base /path/to/maestro_midi/ \
    --composer Chopin \
    --evaluate checkpoints/Chopin_epoch500.pt
```

Evaluate a checkpoint without triggering a training event:

```
python composer_melody_trainer.py \
    --meta_csv /path/to/maestro_midi_metadata.csv \
    --midi_base /path/to/maestro_midi/ \
    --composer Chopin \
    --evaluate checkpoints/Chopin_epoch500.pt
```

Model Training: ```composer_melody_trainer.py```

Song generation with trained model: ```full_song_generator.py```



### Big Changes

Too many issues to list, with the previous approach. Mostly revolving around - here's a shocker - clean training data. 
New approach...

- Use .krn
- Heuristics on .krn
- - asdf
  - asdf
  - asdf

- prepare_phrase_dataset.py # generates the phrase datasets from the .krn

$ python3 prepare_phrase_dataset.py ../Utilities/beethoven_quartets/krn

- train_phrase_transformer.py # uses the datasets for training

$ python3 train_phrase_transformer.py phrase_dataset_long.json

- generate_phrase.py # uses the trained model for phrase generation

$ python3 generate_phrase.py phrase_dataset_long.json 32  

  
