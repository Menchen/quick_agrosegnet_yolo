# Quick agrosegnet YOLO trainer


This repo is a quick compilation vibe coded scripts that train shadow segmentation model using YOLO for agrosegnet https://huggingface.co/datasets/Menchen/AgroSegNet

## Usage


First install `pixi` if not installed `curl -fsSL https://pixi.sh/install.sh | sh`


```bash
pixi install # install dependencies
pixi run python prep.py # download dataset and split into YOLO format
```

By default, `cuda>=13.0` supported driver is required, and `default-tiny` subset is downloaded, ~3GB, 12500 pair of images. Could be changed in `prep.py` to `default` for full dataset, 50,000 pair of images .


After installing the dependencies, there's 3 command, each one for one task. Feel free to change the model type and best model path (YOLO generate new folder for each training in `runs` folders)

The best model needs to be manually changed for `evaluate.py` and `visual_evaluate.py` to use the new model.

```bash
pixi run python train.py # train the model, modify to change parameters like model, epochs...
pixi run python evaluate.py # evaluate the test split, show accuracy and BER (Balance Error Rate used for shadow model metric)
pixi run python visual_evaluate.py # evaluate and show both gt and prediction, as image in visual_evals folder
```
