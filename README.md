# BitumenGrader

BitumenGrader is a desktop app that trains a small CNN on labelled froth
photos and then predicts Water, Solids, and Bitumen percentages from new
images. You load a labels table, match it to a photo folder, train (or
continue) a model, and grade photos one at a time or in a batch. The
default architecture is a compact CNN trained from scratch; ImageNet
transfer (ResNet50 / VGG16) is optional.

## Setup

You need **Python 3.9+** and a few GB of disk space (PyTorch is large).

```bash
cd BitumenGrader
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The window has three pages: **Train**, **Grade**, and **Models**.

1. **Train** — load a labels file (`Image`, `Pan`, `Water`, `Solids`,
   `Bitumen`) and a photo folder, then start training. Sample photos are in
   `BitumenImagesFlotation/`. Saved models go in `models/`.
2. **Models** — load, retrain, or delete a saved model.
3. **Grade** — drop photos onto the queue and run the active model.

Full walkthrough, settings, and troubleshooting: `USER_GUIDE.txt`.
