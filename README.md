# BitumenGrader

Desktop app that trains a CNN to predict Water, Solids, and Bitumen % from froth photos. The default model is a compact CNN trained from scratch; ImageNet transfer (ResNet50 / VGG16) is optional. Train or continue a saved model on a new dataset, grade new photos, and manage saved models.

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

See [USER_GUIDE.txt](USER_GUIDE.txt) for grading, training, settings, and troubleshooting.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/smoke_test.py -v
```

## Package a desktop build

Requires the dev extras (PyInstaller). From the repo root:

```bash
pip install -r requirements-dev.txt
pyinstaller BitumenGrader.spec
```

The folder `dist/BitumenGrader/` is the runnable app (`BitumenGrader.app` on macOS). PyTorch makes the bundle large. Sample photos in `BitumenImagesFlotation/` and checkpoints in `models/` are not copied into the build; a packaged app saves new models in the OS application-data folder.
