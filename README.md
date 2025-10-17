# IMDb Sentiment Analysis: Full Fine-tuning vs LoRA Comparison

A PyTorch implementation of sentiment analysis on the IMDb movie review dataset, comparing two fine-tuning strategies: Full Fine-tuning and LoRA (Low-Rank Adaptation) for parameter efficiency.

## Project Structure

```
.
├── main.py               # Entry point with command-line arguments
├── train_eval.py         # Training, evaluation, and error analysis logic
├── model_setup.py        # Model initialization (full/LoRA fine-tuning)
├── data_processing.py    # IMDb dataset loading and preprocessing
├── visualization.py      # Training curves, confusion matrix visualization
└── requirements.txt      # Project dependencies
```

## Features

- Supports two fine-tuning strategies:
  - Full fine-tuning (updates all model parameters)
  - LoRA fine-tuning (updates only low-rank matrix parameters)
- Automatic tracking of training loss and evaluation metrics (accuracy, F1-score)
- Generates visualizations (training curves, confusion matrix)
- Analyzes and saves misclassified examples
- Command-line interface for easy experiment configuration

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers & Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- Scikit-learn, Matplotlib, Seaborn, Pandas

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python main.py --strategy [full|lora] [options]
```

### Parameters

| Parameter    | Type    | Default | Description                                  |
|--------------|---------|---------|----------------------------------------------|
| --strategy   | str     | full    | Fine-tuning strategy: "full" or "lora"       |
| --epochs     | int     | 3       | Number of training epochs                    |
| --batch_size | int     | 16      | Training batch size                          |
| --no_analyze | flag    | -       | Disable error case analysis (if set)         |

### Examples

1. Run full fine-tuning with default parameters:
```bash
python main.py --strategy full
```

2. Run LoRA fine-tuning with custom parameters:
```bash
python main.py --strategy lora --epochs 5 --batch_size 32
```

3. Run full fine-tuning without error analysis:
```bash
python main.py --strategy full --no_analyze
```

## Output

All results are saved in `./imdb_finetune_results` directory, organized by strategy:

- `./imdb_finetune_results/[strategy]/`: Model checkpoints and training logs
- `./imdb_finetune_results/[strategy]/plots/`: Visualizations
  - Training/validation loss curves
  - Accuracy and F1-score curves
  - Confusion matrix
- `./imdb_finetune_results/[strategy]/error_cases_[strategy].csv`: Misclassified examples (text, true label, predicted label)

## Implementation Details

- **Data Processing**: Uses Hugging Face Datasets to load IMDb dataset, tokenizes text with DistilBERT tokenizer
- **Model Initialization**:
  - Full fine-tuning: Loads pre-trained DistilBERT with all parameters trainable
  - LoRA fine-tuning: Injects LoRA adapters into attention and feed-forward layers (via PEFT library)
- **Training**: Uses Hugging Face Trainer with callback for metric tracking
- **Evaluation**: Computes accuracy and F1-score, generates confusion matrix, and analyzes misclassified examples

## Expected Results

LoRA fine-tuning typically shows:
- Significantly fewer trainable parameters (90%+ reduction compared to full fine-tuning)
- Faster training and lower memory usage
- Comparable performance to full fine-tuning with proper configuration

## Dataset & Model

- **Dataset**: IMDb movie reviews (automatically downloaded via `datasets` library)
- **Base Model**: DistilBERT-base-uncased (automatically downloaded via `transformers` library)
