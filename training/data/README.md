# Data directory

Place datasets here. By default, the training script looks for `training/data/creditcard.csv` (the Kaggle "Credit Card Fraud Detection" dataset). If not found, it will train on a synthetic dummy dataset so the pipeline remains runnable without external data.

## Getting creditcard.csv from Kaggle

1. Create a Kaggle account and accept the dataset terms: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Install the Kaggle CLI and authenticate:

```bash
pip install kaggle
# Put your API credentials at ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY
```

3. Download the file into this folder:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -f creditcard.csv -p training/data --force
unzip -o training/data/creditcard.csv.zip -d training/data
# Ensure the final file path is training/data/creditcard.csv
```

Alternatively, run the helper script:

```bash
python training/data/download_creditcard.py
```
