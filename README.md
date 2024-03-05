# Suicidality History Classifier
A deep learning method to detect personal and family history of suicidal thoughts and behaviors.

This repository maintains the code for detection of personal and family history of suicidal thoughts and behaviours in clinical notes. Pre-trained Bio_Clinical_BERT and GatorTron BERT models are implemented for fine-tunning (using annotated corpus) and classification. 

## 1. Clone/download repository to your local machine
    cd (change directory) to location of cloned repository

## 2. Set up a virtual python environment
    Run the command python3 -m venv stb_env
    Run the command source stb_env/bin/activate in terminal
    Run the command deactivate in terminal.

## 3. Install necessary packages under the virtual environment

## 4. Edits python scripts for input/output database/file information using an IDE or text editor

## 5. Enter the virtual environment with the following command: source stb_env/bin/activate
    Run various python scripts:
    Example Command: python fsh_BioClinicalBERT_classifier.py
    Example Command: python fsh_GatorTron_classifier.py
    Example Command: python sh_BioClinicalBERT_classifier.py
    Example Command: python sh_GatorTron_classifier.py
