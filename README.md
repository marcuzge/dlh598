# DLH598 - DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology

## Purpose

An attempt to reproduce the data preprocessing pipeline, the LightGBM model, the GPBoost model with LSTM post-processing, and the baseline results presented in the "DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology" paper.

## Background

Wang, K., Yang, J., Shetty, A., & Dunn, J. (2024). DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology (version 1.0.0). PhysioNet. https://doi.org/10.13026/62an-cb28

Original Github repository: https://github.com/WillKeWang/DREAMT_FE

## Setup

1. Clone this repository.
2. Create a Conda environment from `.yml` file.
```
conda env create --file environment.yml
```

## How to run
1. Download the dataset from https://physionet.org/content/dreamt/2.0.0/
2. Update `feature_engineering.py`'s `data_folder` path to `data_64hz\` from step 1.
3. Run the feature_engineering script using:
```
 python feature_engineering.py
```
Or you can use the existing pre-populated features_df in `\dataset_sample\features_df`
4. Run the script to calcute quality score:
```
 python calculate_quality_score.py
```
5. Run the eval pipeline:
```
 python main.py
```
Or you can run the 5-fold cross-validation using `python main_cv.py`

## Directory Structure

The main components of the project pipeline includes: 
* Perform preprocessing and feature enginering on the data
* Training models for classification

```bash
.
├── dataset_sample
    └── features_df
        └── SID_domain_features.csv
    └── E4_aggregate_subsample
        └── subsampled_SID_whole_df.csv
    └── participant_info.csv
├── results
│   └── quality_score_per_subject.csv
├── read_raw_e4.py
├── calculate_quality_score.py
├── feature_engineering.py
├── datasets.py
├── models.py
├── main.py
├── lstm_llm.py
└── utils.py

```

## Description

*   `dataset_sample`: A directory housing a sample dataset, including a folder with feature-engineered data for each participant, a file containing participant details, and downsampled raw signal data per participant.
*   `features_df`: A directory storing the files containing the calculated feature-engineered data.
*   `sid_domain_features_df.csv`: A CSV file containing features derived from the raw Empatica E4 data collected during the study.
*   `participant_info.csv`: A CSV file containing essential information about the participants.
*   `quality_score_per_subject.csv`: A file providing a summary of the artifact percentage in each subject's data, computed from the features dataset `sid_domain_features_df.csv`.
*   `read_raw_e4.py`: A module for reading raw Empatica E4 data, alongside sleep stage labels and sleep reports. It generates a dataframe that time-aligns the Empatica E4 data with sleep stages and sleep performance metrics, such as the Apnea-Hypopnea Index.
*   `feature_engineering.py`: A module that takes the processed data from `read_raw_e4.py` and applies feature engineering techniques. The resulting data is then saved in the `feature_df` directory within the `data` folder.
*   `datasets.py`: A module responsible for loading the feature-engineered data from `feature_df`, performing data cleaning and resampling. The processed data is then partitioned into training, testing, and validation sets.
*   `models.py`: A module used to construct, train, and evaluate the models using the datasets prepared by `datasets.py`. It provides performance metrics and a confusion matrix as output.
*   `lstm_llm.py`: A sample lstm implementation created by LLM.
*   `main.py`: The primary script that orchestrates the entire workflow, including data loading, cleaning, splitting, model creation, training, testing, and evaluation.
*   `utils.py`: A script containing a collection of utility functions supporting the processes of data handling, cleaning, splitting, model development, training, testing, and evaluation.