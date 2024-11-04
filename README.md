# Resume Screening using Neural Network

This project is a Resume Screening system that uses Artificial Neural Networks to classify resumes based on predefined categories. The aim is to automate the initial phase of candidate screening, making the recruitment process more efficient. This system reads resume data, preprocesses it, and uses machine learning to categorize candidates.

## Features
- Resume categorization based on content analysis
- ANN model training for high classification accuracy
- GUI for user-friendly interaction
- Model saved using Joblib for easy deployment
# Project Flowchart
![Project Flowchart](https://github.com/sujitmahapatra/Resume-Screening-using-ANN/blob/85445a766ff99867063ba6a26df963d25d0148b1/resume%20screening%20flowchart.png)

## Libraries Used

The project uses the following Python libraries:

- **Numpy** and **Pandas**: For data handling and manipulation.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Sklearn**: Used for machine learning models, metrics, and data preprocessing, including:
  - `MultinomialNB`: Naive Bayes classifier.
  - `OneVsRestClassifier`: For handling multiple classes.
  - `KNeighborsClassifier`: An additional classifier for comparison.
  - `TfidfVectorizer`: For text vectorization.
  - `LabelEncoder`, `train_test_split`: For preprocessing.
- **Scipy**: Provides sparse matrix functionality to optimize memory usage.
- **WordCloud**: For generating word clouds to visualize common resume terms.

# Model Training
![model](https://github.com/sujitmahapatra/Resume-Screening-using-ANN/blob/85445a766ff99867063ba6a26df963d25d0148b1/model%20training.png)
# OUTPUT
![output](https://github.com/sujitmahapatra/Resume-Screening-using-ANN/blob/85445a766ff99867063ba6a26df963d25d0148b1/output.png)

## Usage
- Preprocess the Data: Load the dataset and preprocess it with LabelEncoder, TfidfVectorizer, and other tools to handle text data.
- Train the Model: Use the ANN model to train on labeled resume data, adjusting hyperparameters as needed.
- Evaluate the Model: Evaluate the model’s performance using metrics like accuracy and confusion matrix.
- Save the Model: Save the trained model using Joblib for later use.
