from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create the Flask application
app = Flask(__name__)

# Load the dataset
data_url = "https://raw.githubusercontent.com/ARubiato/lbo-faq-data/main/Live_blood_analysis_training_co_data_updated.csv"
data = pd.read_csv(data_url)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Handling missing values
data.dropna(inplace=True)

# Assuming the columns are 'Question' and 'Answer'
X = data['Question']
y = data['Answer']

# Tokenization and vectorization
vectorizer = CountVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Train the model
classifier = MultinomialNB()
classifier.fit(X_vectors, y)

@app.route('/predict', methods=['POST'])
def predict_answer():
    # Get the question from the request
    question = request.json['question']

    # Vectorize the question
    question_vector = vectorizer.transform([question])

    # Predict the answer
    predicted_answer = classifier.predict(question_vector)[0]

    # Return the predicted answer as JSON
    return jsonify({'predicted_answer': predicted_answer})

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
