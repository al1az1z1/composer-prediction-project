# Composer and Genre Classification using Deep Learning

This project is part of the A511 Neural Networks course. We aim to classify MIDI music files by **composer** and optionally **genre**, using two deep learning approaches: a Recurrent Neural Network (RNN) with LSTM and a Convolutional Neural Network (CNN).

## Project Structure


project/  
├── data/ # Raw and processed data  
│ ├── midi/ # Original MIDI files  
│ ├── spectrograms/ # Audio converted to spectrograms (for CNN)  
│ └── note_sequences/ # Sequences extracted from MIDI (for RNN)
├── models/ # Raw and processed data
| ├── CNN/ # Saved CNN Models 
│ ├── RNN/ # Saved LSTM Models     
├── notebooks/ # Jupyter notebooks for EDA and training  
│ ├── EDA.ipynb  
│ ├── RNN_training.ipynb  
│ ├── CNN_training.ipynb  
├── preprocessing/ # Scripts for data conversion  
│ ├── audio_to_spectrogram.py
│ └── midi_to_sequence.ipynb  
├── reports/ # Final report  
│ └── Project_Report-TeamX.pdf  
├── README.md # This file  
└── requirements.txt # Project dependencies  


## Objectives

- Convert MIDI files into usable inputs for both LSTM and CNN models
- Train and evaluate both models on the dataset
- Compare the performance of both models
- Generate insights for genre/composer classification using deep learning


## Target Variable

The primary target variable in this project is the **composer** of each MIDI file.  
The dataset contains works from the following composers:

- Bach 
- Beethoven
- Chopin
- Mozart

## Models

- `CNN`: Spectrogram-based image classification model
- `RNN`: LSTM-based sequence model using MIDI note sequences


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


## Team Members

- **Shaun Friedman**
  - GitHub: [shaun-friedman](https://github.com/shaun-friedman)
  - LinkedIn: [Shaun Friedman](https://www.linkedin.com/in/shaun-friedman/)
- **Ali Azizi** 
  - GitHub: [al1az1z1](https://github.com/al1az1z1)
  - LinkedIn: [Ali Azizi](https://www.linkedin.com/in/al1az1z1)

---


## License & Academic Use

This project was developed as part of the course **AAI 511 – Neural Networks and Deep Learning** at the **University of San Diego**.  
It is intended for educational purposes only and is released under the **MIT License**.  
All datasets are publicly available and sourced from Kaggle. Code contributions follow the [PEP 8 Style Guide](https://peps.python.org/pep-0008/).
