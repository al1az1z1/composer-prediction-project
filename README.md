# Composer and Genre Classification using Deep Learning

This project is part of the A511 Neural Networks course. We aim to classify MIDI music files by **composer** and optionally **genre**, using two deep learning approaches: a Recurrent Neural Network (RNN) with LSTM and a Convolutional Neural Network (CNN).

## Project Structure


project/  
├── data/ # Raw and processed data  
│ ├── midi/ # Original MIDI files  
│ ├── spectrograms/ # Audio converted to spectrograms (for CNN)  
│ └── note_sequences/ # Sequences extracted from MIDI (for RNN)  
├── preprocessing/ # Scripts for data conversion  
│ ├── midi_to_sequence.py  
│ └── audio_to_spectrogram.py  
├── notebooks/ # Jupyter notebooks for EDA and training  
│ ├── EDA.ipynb  
│ ├── RNN_training.ipynb  
│ ├── CNN_training.ipynb  
│ └── comparison.ipynb  
├── reports/ # Final report  
│ └── Project_Report-TeamX.pdf  
├── README.md # This file  
└── requirements.txt # Project dependencies  


## Objectives

- Convert MIDI files into usable inputs for both LSTM and CNN models
- Train and evaluate both models on the dataset
- Compare the performance of both models
- Generate insights for genre/composer classification using deep learning

## Models

- `RNN`: LSTM-based sequence model using MIDI note sequences
- `CNN`: Spectrogram-based image classification model

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/al1az1z1/composer-prediction-project.git
   cd composer-prediction-project/project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks**:
   - Start with preprocessing scripts or notebooks
   - Train and evaluate models using the provided training notebooks

---

### Dependencies (`requirements.txt`)

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
librosa
jupyter

Install them using:
```bash
pip install -r requirements.txt
```

---

## License

This project is for educational purposes under the A511 course. Open for academic use.

