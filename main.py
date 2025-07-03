from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv("augmented_chatbot_dataset.csv")
questions = df['Question'].astype(str).tolist()
answers = df['Answer'].astype(str).tolist()

# Load model & encode once
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

question_embeddings = model.encode(questions)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    user_embedding = model.encode([user_input])

    similarity = cosine_similarity(user_embedding, question_embeddings)
    idx = similarity.argmax()
    score = similarity[0][idx]

    print(f"Input: {user_input}, Best match score: {score:.2f}, Index: {idx}")

    if score < 0.4:
        return jsonify({"response": "Hmm, I didn’t understand that. Can you rephrase?"})

    return jsonify({"response": answers[idx]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
