\# Token Classification for Feedback Prize NLP Task



This repository contains code for training and inference of a token classification model based on Google's BigBird transformer architecture. The task is to identify discourse elements in student essays, such as claims, evidence, and rebuttals, by tagging tokens with BIO labels.



---



\## Table of Contents



\- \[Project Overview](#project-overview)

\- \[Requirements](#requirements)

\- \[Setup](#setup)

\- \[Data Preparation](#data-preparation)

\- \[Training](#training)

\- \[Inference](#inference)

\- \[Visualization](#visualization)

\- \[Evaluation](#evaluation)

\- \[Notes](#notes)



---



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



sample\_row = valid\_data.loc\[0]

sample\_text = sample\_row\['text']

sample\_df = pd.DataFrame({'text': \[sample\_text], 'entities': \[None]})

sample\_dataset = dataset(sample\_df, tokenizer, config\['max\_length'], get\_wids=True)

sample\_loader = DataLoader(sample\_dataset, batch\_size=1, shuffle=False)



model.eval()

with torch.no\_grad():

&nbsp;   for batch in sample\_loader:

&nbsp;       preds = inference(batch)



sample\_spans = preds\_to\_spans(preds\[0])

display(HTML(highlight\_text(sample\_text, sample\_spans)))





