# ASL Alphabet Interpreter

A real-time American Sign Language (ASL) alphabet interpreter using computer vision and deep learning. This project uses a webcam to capture hand signs and translates them into text, with optional translation capabilities.

## Features

- Real-time ASL alphabet recognition using webcam input
- Deep learning model based on ResNet architecture
- Support for A-Z alphabet signs
- Optional translation of spelled-out words
- Real-time visualization of recognized signs

## Prerequisites

- Python 3.8 or higher
- Webcam
- CUDA-capable GPU (recommended for better performance)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd asl-interpreter
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

1. Download the ASL alphabet dataset from Kaggle (link to be provided)
2. Extract the dataset into the `data/` directory
3. The expected structure should be:
```
data/
    train/
        A/
        B/
        ...
    test/
        A/
        B/
        ...
```

## Usage

### Training the Model

To train the model on your dataset:
```bash
python src/train.py
```

### Running the Application

To start the real-time ASL interpreter:
```bash
python main.py
```

## Project Structure

```
.
├── data/               # Dataset directory
├── src/
│   ├── data_loading.py    # Dataset loading and preprocessing
│   ├── model.py          # Model architecture definition
│   ├── train.py          # Training script
│   ├── inference.py      # Inference utilities
│   ├── translate.py      # Translation functionality
│   └── webcam_capture.py # Webcam handling
├── main.py             # Main application entry point
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Notes

- The model's performance depends on lighting conditions and camera quality
- Real-time performance may vary based on hardware specifications
- The translation feature requires an internet connection
- The model works best with clear, well-lit hand signs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 