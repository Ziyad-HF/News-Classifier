# News Classifier

## Overview

The News Classifier project aims to distinguish between fake and real news articles using machine learning models. The dataset includes text from news articles along with their subjects and titles. This project involves data preprocessing, feature extraction using TF-IDF, and training various machine learning models to classify the news.

## Project Structure

The project consists of two main notebooks:

1. **Preprocessing Notebook**: This notebook handles the data cleaning and preprocessing tasks.
2. **NLP & Modeling Notebook**: This notebook performs text preprocessing, feature extraction, model training, hyperparameter tuning, and evaluation.

## Requirements

To run this project, you need the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- re
- textblob
- nltk
- scikit-learn
- joblib

You can install the required packages using the following command:

```bash
pip install pandas numpy matplotlib seaborn textblob nltk scikit-learn joblib
```

## Data

The project uses a dataset containing news articles with the following columns:
- `ID`: Unique identifier for each article
- `title`: Title of the article
- `subject`: Subject of the article
- `text`: Full text of the article
- `date`: Date the article was published
- `class`: Class label indicating whether the news is real or fake

## Notebooks

### Preprocessing Notebook

This notebook handles data cleaning and preprocessing:

1. **Importing necessary libraries**
2. **Loading the data**: Load the train and test datasets.
3. **Exploring the data**: Display the first few rows, summary statistics, and info about the datasets.
4. **Preprocessing the data**:
   - Handling duplicates
   - Handling missing values
   - Handling date data
5. **Saving the preprocessed data**: Save the cleaned train and test datasets.

### NLP & Modeling Notebook

This notebook performs text preprocessing, feature extraction, and model training:

1. **Imports**: Import necessary libraries.
2. **Text preprocessing**:
   - Load the cleaned train and test datasets.
   - Define a text cleaning function that includes tokenization and lemmatization.
   - Apply the text cleaning function to the datasets.
   - Combine the cleaned text, subject, and title into a single column.
3. **TF-IDF Vectorization**: Convert the combined text fields into TF-IDF features.
4. **Modeling**:
   - Split the dataset into training and validation sets.
   - Train and evaluate models using GridSearchCV for hyperparameter tuning:
     - Decision Tree Classifier
     - Naive Bayes Classifier
   - Save the predictions and best models.

## Results

The best model achieved is the Decision Tree model, which attained an accuracy of 0.99925 on the test data and a perfect accuracy of 1.0 on the training data. This exceptional performance enabled our team, "Attack on Tensors," to secure the first place in the Kaggle competition for this dataset. You can view the competition details [here](https://www.kaggle.com/competitions/gdsc-ml-workshop-final-project).

## Usage

1. **Preprocess the data**:
   - Run the Preprocessing Notebook to clean the train and test datasets.

2. **Train and evaluate models**:
   - Run the NLP & Modeling Notebook to preprocess the text data, perform TF-IDF vectorization, train the models, and save the predictions and best models.


## Conclusion

This project provides a comprehensive approach to classifying news articles using machine learning. The preprocessing steps and model training processes are well-documented and can be easily replicated for other similar datasets.

## Contributors

- [Ziyad El-Fayoumy](https://github.com/Zoz-HF)
- [Hesham Hatem](https://github.com/Hesham942)
- [Engy Sherif](https://github.com/EngySherif)
