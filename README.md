# BitumenGrader

Desktop app that trains a CNN to predict Water, Solids, and Bitumen % from froth photos. The default model is a compact CNN trained from scratch; ImageNet transfer (ResNet50 / VGG16) is optional. You can import/edit images, train or continue a saved model on a new dataset, grade new photos, and manage saved models.

## Install

```bash
git clone <repository-url> BitumenGrader
cd BitumenGrader
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

## Docs

See [USER_GUIDE.txt](USER_GUIDE.txt) for install, grading, training, settings, and troubleshooting.

## Tests

```bash
python -m pytest tests/smoke_test.py -v
```
