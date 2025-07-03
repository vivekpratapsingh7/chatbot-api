from flask import Flask, request, jsonify
import pandas as pd
from fuzzywuzzy import fuzz

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv("augmented_chatbot_dataset.csv")
questions = df['Question'].astype(str).tolist()
answers = df['Answer'].astype(str).tolist()

# Match user input with best question
def get_response(user_input):
    best_score = 0
    best_idx = -1
    for idx, question in enumerate(questions):
        score = fuzz.ratio(user_input.lower(), question.lower())
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score < 60:
        return "I'm not sure I understood that. Can you rephrase?"
    return answers[best_idx]

# API route
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"response": response})

# For Railway: use PORT env variable
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
