# Hate Speech Detection Project

This repository contains coursework for building a hate speech detection model. It currently includes a Jupyter notebook that explores fine-tuning a BERT pipeline and evaluating performance on a labeled dataset.

## Repository contents
- `Bert_pipeline.ipynb`: main notebook for data preparation, model fine-tuning, and evaluation.

## Requirements
- Python 3.9+
- Jupyter Notebook or JupyterLab
- PyTorch and Transformers (Hugging Face)

## How to run
1. Install dependencies (example using `pip`):
   ```bash
   pip install torch transformers jupyter
   ```
2. Launch the notebook server from the repository root:
   ```bash
   jupyter notebook
   ```
3. Open `Bert_pipeline.ipynb` in the browser, run the cells sequentially, and update any dataset paths to match your local environment.

## Notes
- Ensure you have a suitable GPU environment when fine-tuning BERT for reasonable training times.
- If you extend the project, consider adding data preprocessing scripts and exporting trained models for easier reuse.
