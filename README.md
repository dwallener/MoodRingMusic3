# MoodRingMusic3
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
