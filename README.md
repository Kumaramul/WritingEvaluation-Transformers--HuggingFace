\# Token Classification for Feedback Prize NLP Task



This repository contains code for training and inference of a token classification model based on Google's BigBird transformer architecture. The task is to identify discourse elements in student essays, such as claims, evidence, and rebuttals, by tagging tokens with BIO labels.





\## Project Overview



This project uses the BigBird transformer model for token classification to detect different discourse types in text data. It supports:



\- Tokenizing text data with the BigBird tokenizer

\- Preparing datasets with BIO tagging scheme

\- Training with configurable learning rates and batch sizes

\- Validating and scoring predictions based on the Kaggle Feedback Prize competition metric

\- Visualizing predicted entity spans with highlights



---



\## Requirements



\- Python 3.8+

\- PyTorch

\- Transformers (HuggingFace)

\- Pandas

\- NumPy

\- scikit-learn

\- tqdm

\- Jupyter Notebook (optional, for visualization)







./data/

&nbsp;   train.csv

&nbsp;   train/

&nbsp;       \*.txt

&nbsp;   test/

&nbsp;       \*.txt







