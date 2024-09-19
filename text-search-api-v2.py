from flask import Flask, request, jsonify
import pandas as pd
from scipy import spatial
import os
import openai
import uuid
import pymongo
from pymongo import MongoClient
from datetime import datetime

# Initialize the Flask app
app = Flask(__name__)

# Set OpenAI API key (ensure the key is stored as an environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()

# Set up MongoDB client
MONGO_URI = os.getenv("MONGO_URI")  # Ensure your MongoDB URI is set in the environment variables
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['mydatabase']  # Replace 'mydatabase' with your database name
embeddings_collection = db['embeddings']  # Collection to store embeddings

# Embedding creation function
def create_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Add a new text and its embedding to MongoDB
def add_text_to_db(session_id, text):
    text_embedding = create_embedding(text)
    document = {
        'session_id': session_id,
        'text': text,
        'embedding': text_embedding,
        'timestamp': datetime.utcnow()
    }
    embeddings_collection.insert_one(document)
    return document

# Function to rank texts by relatedness
def texts_ranked_by_relatedness(query, session_id, top_n=100):
    query_embedding = create_embedding(query)
    
    # Fetch all embeddings for the session_id
    cursor = embeddings_collection.find({'session_id': session_id})
    
    texts = []
    embeddings = []
    for doc in cursor:
        texts.append(doc['text'])
        embeddings.append(doc['embedding'])
    
    # Compute similarities
    relatedness_scores = [1 - spatial.distance.cosine(query_embedding, emb) for emb in embeddings]
    
    # Combine texts and scores
    texts_and_scores = list(zip(texts, relatedness_scores))
    
    # Sort by scores
    texts_and_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N
    top_texts_and_scores = texts_and_scores[:top_n]
    
    top_texts = [t[0] for t in top_texts_and_scores]
    top_scores = [t[1] for t in top_texts_and_scores]
    
    return top_texts, top_scores

# Flask routes

# Route to add text for a specific session
@app.route('/add_text', methods=['POST'])
def add_text():
    data = request.json
    session_id = data.get('session_id')
    text = data.get('text')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Add text to MongoDB
    document = add_text_to_db(session_id, text)
    
    return jsonify({'message': 'Text added successfully', 'text': text}), 200

# Route to search for relevant texts for a specific session
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    top_n = data.get('top_n', 100)  # Default to 100 if not specified
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check if there is data for the session_id
    if embeddings_collection.count_documents({'session_id': session_id}) == 0:
        return jsonify({'error': f'No data found for session_id {session_id}'}), 404
    
    # Perform the search
    top_texts, top_scores = texts_ranked_by_relatedness(query, session_id, top_n)
    
    return jsonify({'top_texts': top_texts, 'top_scores': top_scores}), 200

# Route to get session data
@app.route('/get_session_data', methods=['GET'])
def get_session_data():
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    cursor = embeddings_collection.find({'session_id': session_id})
    data = []
    for doc in cursor:
        data.append({
            'text': doc['text'],
            'timestamp': doc.get('timestamp')
        })
    
    if not data:
        return jsonify({'error': 'No data found for session_id'}), 404
    
    return jsonify(data), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
