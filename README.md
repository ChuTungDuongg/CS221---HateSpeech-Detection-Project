# Hate Speech Detection Project

This repository contains coursework for building a hate speech detection model using a fine-tuned BERT classifier. The included notebook walks through loading a labeled dataset, preprocessing text, training, and evaluating the model.

## Repository contents
- `Bert_pipeline.ipynb`: end-to-end notebook covering data preparation, tokenization, model fine-tuning, and evaluation.

## Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab
- GPU is recommended for training; CPU works for experimentation but will be slower.

## Setup
1. (Recommended) Create and activate a virtual environment.
2. Install dependencies with pip:
   ```bash
   pip install torch transformers datasets scikit-learn jupyter
   ```
   Choose the appropriate `torch` build for your hardware (CPU or CUDA).

## Data expectations
- The notebook assumes a labeled dataset with at least a `text` column and a corresponding numeric or categorical `label`.
- Update any dataset paths in the notebook to point to your local files.
- If your labels are not already numeric, map them to integers before training (see the preprocessing section of the notebook for guidance).

## Running the notebook
1. Start Jupyter from the repository root:
   ```bash
   jupyter notebook
   ```
2. Open `Bert_pipeline.ipynb` in your browser.
3. Run the cells in order, adjusting file paths, label mappings, and hyperparameters (e.g., batch size, learning rate, number of epochs) to fit your environment.

## Evaluation and experimentation
- The notebook reports common classification metrics such as accuracy, precision, recall, and F1-score.
- To iterate quickly, try training on a subset of the data or reducing the number of epochs.
- Save trained models and tokenizer artifacts to reuse them without retraining (see the final cells for an example workflow).

## Next steps
- Add dataset download/processing scripts for repeatable experiments.
- Log metrics to a tracking tool (e.g., TensorBoard) for easier comparison across runs.
- Export the fine-tuned model for downstream applications such as content moderation pipelines.
