# Twitter Sentiment Analysis using Logistic Regression

## Overview
This project performs sentiment analysis on tweets using a Logistic Regression model. The dataset used for training and evaluation is the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset from Kaggle, which contains labeled tweets for binary sentiment classification (positive or negative). The model achieves an accuracy of **77.67%** on the test data.

## Dataset
The Sentiment140 dataset consists of 1.6 million tweets with the following columns:
- `target`: Sentiment label (0 = Negative, 4 = Positive)
- `ids`: Unique ID for the tweet
- `date`: Timestamp of the tweet
- `flag`: Query flag (unused in this project)
- `user`: Username of the author
- `text`: The actual tweet content

For this project, only the `target` and `text` columns were used.

## Project Workflow
### 1. Data Preprocessing
To prepare the data for training, the following preprocessing steps were applied:
- Removed unnecessary columns (`ids`, `date`, `flag`, `user`)
- Converted labels (4 → 1 for positive, 0 remains negative)
- Lowercased all text
- Removed special characters, punctuation, and numbers
- Removed stopwords using NLTK
- Tokenized and stemmed words using PorterStemmer
- Transformed text into numerical representation using TF-IDF vectorization

### 2. Model Training
- Split the dataset into **training** and **testing** sets (80-20 split)
- Used **TF-IDF vectorization** to convert text data into numerical form
- Trained a **Logistic Regression** model using `sklearn.linear_model.LogisticRegression`
- Optimized hyperparameters using Grid Search (if applicable)

### 3. Model Evaluation
- Evaluated the model using accuracy, precision, recall, and F1-score
- Achieved a test accuracy of **77.67%**

## Installation & Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn nltk
```

### Running the Project
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```
2. Download the Sentiment140 dataset from Kaggle and place it in the project directory.
3. Run the preprocessing and model training script:
   ```bash
   python train.py
   ```
4. Evaluate the model:
   ```bash
   python evaluate.py
   ```

## Results
The model achieved the following performance on the test set:
- **Accuracy**: 77.67%
- **Precision**: (to be filled based on results)
- **Recall**: (to be filled based on results)
- **F1-score**: (to be filled based on results)

## Future Improvements
- Experiment with different feature extraction methods (e.g., word embeddings like Word2Vec, FastText, or BERT)
- Try deep learning models like LSTMs or Transformers for better performance
- Fine-tune hyperparameters further
- Use more advanced text preprocessing techniques

## License
This project is open-source and available under the MIT License.

