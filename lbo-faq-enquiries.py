# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/ARubiato/lbo-faq-data/main/Live_blood_analysis_training_co_data_updated.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Handling missing values
data.dropna(inplace=True)

# Assuming the columns are 'Question' and 'Answer'
X = data['Question']
y = data['Answer']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and vectorization
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
classifier = MultinomialNB()
classifier.fit(X_train_vectors, y_train)

# Evaluate the model
accuracy = classifier.score(X_test_vectors, y_test)
print("Accuracy:", accuracy)

# # Example prediction
# new_question = ["How often do you run the courses?"]
# new_question_vector = vectorizer.transform(new_question)
# predicted_answer = classifier.predict(new_question_vector)
# print("Predicted Answer:", predicted_answer)


