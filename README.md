# MLP Letter Recognizer

Simple neural network project using a Multilayer Perceptron (MLP) written from scratch in Python.
The program learns to recognize hand-drawn letters **C**, **O** and **T** based on samples provided by the user.

The model is trained on a small custom dataset collected through a drawing interface.

## Features

- MLP neural network implemented **without ML frameworks**
- Custom dataset collection tool
- Simple GUI for drawing and recognition
- Ability to add samples of unknown symbols to improve rejection
- Basic prediction filters to reduce incorrect classifications:
    - reject drawings that are too small (almost empty canvas)
    - reject drawings that are too large (random scribbles)
    - reject predictions with low confidence

## Requirements

**Python 3.x**

Install dependencies:

`pip install -r requirements.txt`

Required libraries:

```
numpy
matplotlib
```

## Project Structure
- `collect.py` - tool for collecting training samples
- `train.py` - trains the MLP model
- `play.py` - GUI for testing the trained model
- `dataset.npz` - saved dataset (generated)
- `model.npz` - trained model weights (generated)

## Usage

### 1. Collect training data

Run:

```
python collect.py
```

#### Draw a symbol and press:
- C – save sample of letter C
- O – save sample of letter O
- T – save sample of letter T
- X – save sample of an unknown symbol
- ESC – save dataset and exit

Thiss will create `dataset.npz`.

### 2. Train the model

```
python train.py
```

This will train the MLP and create `model.npz`.

### 3. Run recognition app

```
python play.py
```

Draw a symbol and press button to see the prediction.

## Notes

The project was tested using the following dataset:
- 10 samples of **C**
- 10 samples of **O**
- 10 samples of **T**
- 10 samples of **other / unknown symbols**