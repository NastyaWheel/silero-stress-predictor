# Silero Stress Predictor — Kaggle Competition Solution

This repository contains my solution for the [Расстановка ударений на русском языке](https://www.kaggle.com/competitions/silero-stress-predictor) Kaggle competition.

The goal of the competition was to predict the stressed syllable in a given Russian word.

## Repository Structure

```text
├── final model/              # Final solution for inference and training
│   ├── model.py             # Model architecture (StressLSTM)
│   ├── train.py             # Training pipeline
│   ├── dataset.py          # Custom PyTorch Dataset and preprocessing
│   ├── main.py             # Full pipeline: training, inference, submission generation
│   └── notebook.ipynb      # Jupyter Notebook for running the final solution

├── reasoning.ipynb          # Notebook with exploratory data analysis, experiments and solution reasoning
├── .gitignore               # Git ignored files
└── README.md                # Project documentation
```

## Approach

The final model is a **custom StressLSTM**. It takes character-level input enriched with three handcrafted features:

- binary vowel mask,
- normalized character position,
- normalized vowel index.

These features are concatenated with trainable embeddings and fed into a BiLSTM. Only vowel positions are used for stress prediction. The model outputs one logit per vowel via a linear layer, and the highest-scoring vowel is selected as the stressed syllable.

## Results

My final solution achieved **0.85977** accuracy on the public leaderboard and **0.8000** accuracy on the private.

## How to run

This solution is intended to be run in a Kaggle Notebook environment. For running the solution:

1. Create a new Kaggle Notebook.

2. In the *Data* tab of your notebook:
  - Add the original competition dataset: [silero-stress-predictor](https://www.kaggle.com/competitions/silero-stress-predictor)
  - Add your custom dataset with code files from the `final model` folder (saved as `.py` files). Name this dataset: `stress` to match the path in the `notebook.ipynb`.

3. Import `notebook.ipynb` from this repository into the Kaggle Notebook environment.

4. Run all cells in the notebook. 

The notebook will automatically train the model and generate a `submission.csv` file for the competition.
