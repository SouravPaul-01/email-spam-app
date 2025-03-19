# SMS Spam Detection Application

## Overview
The SMS Spam Detection Application is a machine learning-based tool designed to classify SMS messages as either spam or legitimate (not spam). Built using Python and Streamlit, this application provides a user-friendly web interface where users can input a text message and receive instant feedback on its classification.

## Features
- **Real-time Analysis**: Instantly classify SMS messages as spam or not spam.
- **User-Friendly Interface**: Simple and intuitive web application built with Streamlit.
- **Natural Language Processing**: Utilizes advanced text processing techniques to analyze message content.

## How It Works

### Technical Process
1. **Text Preprocessing**:
   - Converts the input text to lowercase.
   - Tokenizes the message into individual words.
   - Removes non-alphanumeric characters.
   - Eliminates common stopwords (e.g., "the", "is") and punctuation.
   - Applies stemming to reduce words to their root form (e.g., "running" becomes "run").

2. **Text Vectorization**:
   - Transforms the processed text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
   - This conversion allows machine learning algorithms to understand the text data.

3. **Classification**:
   - Uses a pre-trained Multinomial Naive Bayes model to classify messages.
   - Returns a binary classification: Spam (1) or Not Spam (0).

### The Machine Learning Model
- The model was trained on a dataset of 5,572 SMS messages, each labeled as spam or ham (legitimate).
- Data cleaning removed duplicates, resulting in 5,169 messages for training.
- The dataset is imbalanced, with 12.63% of messages classified as spam.
- Various Naive Bayes classifiers were tested, with Multinomial Naive Bayes showing the best performance.
- The final model achieved an accuracy of 97% with a precision of 0.88 for spam detection.

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SouravPaul-01/sms-spam-detection.git
   cd sms-spam-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the web interface**:
   Open a web browser and go to `http://localhost:3000`.

## Usage
1. Enter an SMS message in the text input field.
2. Click the "Predict" button.
3. View the classification result (Spam or Not Spam).

## Project Structure

## Technology Used
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit

### Data Collection
The SMS Spam Collection dataset was collected from Kaggle, which contains over 5,500 SMS messages labeled as either spam or not spam.
You can access the dataset from [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Data Cleaning and Preprocessing
The data was cleaned by handling null and duplicate values, and the "type" column was label-encoded. The data was then preprocessed by converting the text into tokens, removing special characters, stop words and punctuation, and stemming the data. The data was also converted to lowercase before preprocessing.

### Exploratory Data Analysis
Exploratory Data Analysis was performed to gain insights into the dataset. The count of characters, words, and sentences was calculated for each message. The correlation between variables was also calculated, and visualizations were created using pyplots, bar charts, pie charts, 5 number summaries, and heatmaps. Word clouds were also created for spam and non-spam messages, and the most frequent words in spam texts were visualized.

### Model Building and Selection
Multiple classifier models were tried, including NaiveBayes, random forest, KNN, decision tree, logistic regression, ExtraTreesClassifier, and SVC. The best classifier was chosen based on precision, with a precision of 100% achieved.

### Web Deployment
The model was deployed on the web using Streamlit. The user interface has a simple input box where the user can input a message, and the model will predict whether it is spam or not spam.


## Contributions
Contributions to this project are welcome. If you find any issues or have any suggestions for improvement, please open an issue or a pull request on this repository.


