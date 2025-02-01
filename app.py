from flask import Flask, request, render_template
import joblib
from scipy.sparse import hstack
from sinling import SinhalaTokenizer
import re

app = Flask(__name__)

# Load models and vectorizers
svm_model = joblib.load('./model/svm_model.pkl')
char_vectorizer = joblib.load('./model/char_vectorizer.pkl')
word_vectorizer = joblib.load('./model/word_vectorizer.pkl')

# Initialize Sinhala tokenizer
tokenizer = SinhalaTokenizer()

def is_valid_sinhala(sentence):
    """Check if the input is Sinhala Unicode text."""
    return bool(re.match(r'^[\u0D80-\u0DFF\s]+$', sentence))

# Function to predict noun and case
def predict_case(sentence):
    # Tokenize the sentence to find the noun
    tokens = tokenizer.tokenize(sentence)
    noun = tokens[0] if tokens else "Unknown"  # Default to the first token

    # Transform sentence for prediction
    char_features = char_vectorizer.transform([sentence])
    word_features = word_vectorizer.transform([sentence])
    combined_features = hstack([char_features, word_features])

    # Predict the noun case
    predicted_case = svm_model.predict(combined_features)[0]

    # Special handling for ablative case
    if predicted_case == "Ablative" and len(tokens) > 1:
        # Assume the noun is the second word for ablative cases
        noun = tokens[1]

    return noun, predicted_case

# Home route
@app.route('/')
def index():
    return render_template('index.html', noun=None, case=None, sentence=None, error=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.form['sentence']

    # Validate the input
    if not sentence:
        error_message = "Input cannot be empty. Please enter a Sinhala Unicode sentence."
        return render_template('index.html', error=error_message)
    
    if not is_valid_sinhala(sentence):
        error_message = "Invalid input. Please enter a valid Sinhala Unicode sentence."
        return render_template('index.html', error=error_message)
    
    noun, case = predict_case(sentence)
    return render_template('index.html', noun=noun, case=case, sentence=sentence)

if __name__ == '__main__':
    app.run(debug=True)
