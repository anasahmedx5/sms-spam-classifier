# SMS Spam Classifier

## Project Overview

This project aims to build a classifier that can distinguish between **ham** (non-spam) and **spam** messages using machine learning techniques. The project employs text preprocessing, feature extraction using **TF-IDF**, and machine learning models like **Naive Bayes** and **Support Vector Machine (SVM)** to classify SMS messages as spam or not. The results are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

## Design Overview

### Data Preprocessing
- **Dataset**: SMS Spam Collection dataset (from Kaggle).
- **Feature Selection**:
  - Label: `ham` (0) or `spam` (1)
  - Text: The message content.
- **Text Cleaning**:
  - Lowercased all text.
  - Removed non-alphanumeric characters (e.g., punctuation).
  - Tokenized and cleaned up the text using regular expressions.
- **Label Encoding**: The labels were encoded as `0` for ham and `1` for spam.
- **Feature Extraction**: 
  - **TF-IDF (Term Frequency-Inverse Document Frequency)** was used to convert the cleaned text into numerical features, while removing common stopwords.
  
### Model Building
- **Naive Bayes (MultinomialNB)**: Chosen for its simplicity and efficiency with text data, especially with a bag-of-words representation.
- **Support Vector Machine (SVM)**: Used for its ability to find a hyperplane that best separates the classes, with a linear kernel for simplicity.
- **Train-Test Split**: 80% of the data was used for training, and 20% was reserved for testing.
- **Evaluation Metrics**: 
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix for visualizing the model performance.

### Visualization
- **Word Cloud**: A visualization of the most frequent words in spam messages.
- **Class Distribution Plot**: A count plot showing the distribution of spam and ham messages in the dataset.
- **Confusion Matrix**: For evaluating model performance on the test set.

## Key Components

### Libraries Used
- `pandas`: For data manipulation and preprocessing.
- `numpy`: For numerical operations.
- `matplotlib`, `seaborn`: For visualization and plotting.
- `sklearn`: For TF-IDF vectorization, model training, and evaluation (Naive Bayes, SVM, metrics).
- `wordcloud`: For generating word clouds from text data.

### Process
1. Load and preprocess the dataset.
2. Clean the text data (lowercase, remove punctuation).
3. Encode the labels as 0 for ham and 1 for spam.
4. Apply TF-IDF vectorization to convert the text data into numerical form.
5. Split the dataset into training and testing sets.
6. Train models using **Naive Bayes** and **SVM**.
7. Evaluate the models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Visualize key insights with a word cloud and class distribution plot.

## Algorithms Used
- **Naive Bayes (MultinomialNB)**: A probabilistic model suitable for text classification.
- **Support Vector Machine (SVM)**: A discriminative classifier that finds the optimal decision boundary.
- **TF-IDF**: Feature extraction technique that reflects the importance of words in the dataset.
  
## Results
- **Naive Bayes** achieved an accuracy of **96.77%**. While the model performed well overall, it showed lower recall for spam messages (76%).
- **SVM** performed better with an accuracy of **98.03%** and higher recall for spam messages (87%).
- The confusion matrix and classification report revealed the strengths and weaknesses of both models.
- The word cloud highlighted frequent spam-related terms, aiding in understanding the common patterns in spam messages.

## Conclusion
This project demonstrates the power of machine learning in text classification tasks. By applying both Naive Bayes and Support Vector Machine algorithms, we were able to achieve high accuracy and uncover interesting insights into spam detection. The results can be applied to improve real-time spam filtering systems for SMS services.


