# SniffleCut

AI-powered sniffle sound event detection and automated sniffle removal script for Final Cut Pro's FCPXML video project files.

## Demo

(Enable audio in video player)
https://github.com/user-attachments/assets/e0ae92cc-c888-4a64-bf53-f27850b0f2fa

- Video clip 1: sniffles in audio
- File > Export to XML
- Terminal: run `python sniffle_cut_fcpxml.py`
- Final Cut Pro: File > Import XML
- Video clip 2: sniffle sections automatically cut out from the video

Demo video is in real-time, run on Macbook Pro M3. Optimally the workflow would be a one-click process, but Final Cut Pro does not seem to allow plugins to execute arbitrary scripts or native code - hence the XML steps.

## Background

The sniffle sound event detection is based on the [BEATs](https://arxiv.org/abs/2212.09058) implementation found from https://github.com/microsoft/unilm/tree/master/beats

The pre-trained BEATs model is combined with a [multilayer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron) head, which is fine-tuned to act as a binary "sniffle" / "no sniffle" classifier on the [FluSense](https://dl.acm.org/doi/10.1145/3381014) dataset available e.g. at https://huggingface.co/datasets/vtsouval/flusense/tree/main

With a 80/20 training/validation split of the 1400+ FluSense dataset samples with 6 different classes, an accuracy of 0.985 on the validation set was achieved after 3 epochs on the binary sniffle classifying task. The result is stored in `training_checkpoints/sniffle_head_3epochs.pt`. Further training seemed to result in over-fitting, and worse performance on real-world testing outside of the test data set. (The training can be repeated using the included script `sniffle_detect_and_train.py` by passing the parameter `--mode train`)

The classifier analysis happens in 200ms minimum length windows, but by analyzing in overlapping windows hops, a better temporal resolution of 50ms is achieved.

## Installation

[UV Python package and project manager](https://docs.astral.sh/uv/) setup (installs dependencies and activates a Python virtual environment)

```
uv venv
uv sync
source .venv/bin/activate
```

### Pre-requisites

[FFMpeg](https://www.ffmpeg.org/) should be installed and found in PATH for direct invocation. The FCPXML script extracts the audio from the referenced video clips with FFMpeg.

[Git LFS](https://git-lfs.com/) for checking out the large `.pt` model files.

[UV Python package and project manager](https://docs.astral.sh/uv/)

## Usage

FCPXML example:

```
SniffleCut % python sniffle_cut_fcpxml.py ~/Desktop/FCPXML.fcpxmld/Info.fcpxml 
Processing FCPXML: /Users/mpesonen/Desktop/FCPXML.fcpxmld/Info.fcpxml
Output will be saved to: /Users/mpesonen/Desktop/FCPXML_no_sniffles.fcpxmld/Info.fcpxml

Found 1 asset-clip(s) to process

[1/1] Processing asset-clip...
  Processing clip: DJI_20251229204602_0178_D
    Audio extracted: 14.52s at 16000Hz
    Detecting sniffles...
    Found 4 sniffle segment(s):
      0.850s - 1.100s (prob: 0.96)
      7.900s - 8.100s (prob: 0.92)
      8.550s - 8.750s (prob: 0.97)
      10.900s - 11.150s (prob: 0.92)
    Will split into 5 segment(s)

Modifying FCPXML to remove sniffle segments...
Modified 1 clip(s)
Recalculating offsets for continuity...

✓ Saved modified FCPXML to: /Users/mpesonen/Desktop/FCPXML_no_sniffles.fcpxmld/Info.fcpxml
```

Example of running detection only - outputs JSON to stdout:

```
SniffleCut % python sniffle_detect_and_train.py --verbose --wav sniffle_test.wav
Using device: mps
[{"start": 2.15, "end": 2.4, "probability": 0.94}, {"start": 7.5, "end": 7.7, "probability": 0.71}]
```

Training example:

```
SniffleCut % python sniffle_detect_and_train.py --mode train --epochs 1
Epoch 1/1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 1169/1169 [14:36<00:00,  1.33batch/s, loss=0.0926, acc=0.9647]
Epoch 1: train_loss=0.0926, train_acc=0.9647, val_loss=0.0574, val_acc=0.9808
  -> New best validation loss: 0.0574
Saved best sniffle head (val_loss=0.0574) to sniffle_head.pt
```



