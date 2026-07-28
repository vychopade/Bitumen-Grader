# BitumenGrader

BitumenGrader is a PyQt6 desktop application that trains and runs a convolutional neural network (CNN) to classify photographs of bitumen samples into performance grades (e.g. PG 52-28, PG 64-22), with full workflows for importing/editing images, training models, grading new images, and managing saved models.

## Installation

```bash
git clone <repository-url> BitumenGrader
cd BitumenGrader
pip install -r requirements.txt
```

Then launch the app:

```bash
python main.py
```

## Screenshot

![BitumenGrader screenshot placeholder](assets/screenshot.png)
*(Add a screenshot of the app here, e.g. the Grade Images page with results.)*

## Documentation

For a full walkthrough written for non-technical users -- installation, grading images, training a model, hyperparameter reference, image editing, model management, and troubleshooting -- see [USER_GUIDE.txt](USER_GUIDE.txt).

## Tests

A small offline smoke-test suite covers the ML backend (`BitumenCNN`, `ModelPredictor`, and model save/load round-tripping):

```bash
python -m pytest tests/smoke_test.py -v
```
