from flask import Flask, request, jsonify
import pandas as pd
from scipy import spatial
import os
import openai
import uuid

# Initialize the Flask app
app = Flask(__name__)

# Create a dictionary to store session-specific dataframes
session_data = {}

# Set OpenAI API key (ensure the key is stored as an environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI
client = OpenAI()


# Embedding creation function
def create_embedding(text):

    response = client.embeddings.create(
        input= text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

# Add a new text and its embedding to the dataframe
def add_text_to_dataframe(df, text):
    text_embedding = create_embedding(text)
    new_row = pd.DataFrame([{
        'text': text,
        'embedding': text_embedding
    }])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    return updated_df

# Function to rank texts by relatedness
def texts_ranked_by_relatedness(query, df, top_n=100):
    query_embedding = create_embedding(query)
    
    # Calculate relatedness (cosine similarity)
    relatedness_scores = df['embedding'].apply(lambda emb: 1 - spatial.distance.cosine(query_embedding, emb))
    
    # Sort by relatedness scores
    sorted_df = df.assign(relatedness=relatedness_scores).sort_values(by='relatedness', ascending=False)
    
    top_texts = sorted_df['text'].head(top_n).tolist()
    top_scores = sorted_df['relatedness'].head(top_n).tolist()
    
    return top_texts, top_scores

# Flask routes

# 1. Route to start a new session
@app.route('/start_session', methods=['POST'])
def start_session():
    session_id = str(uuid.uuid4())
    session_data[session_id] = pd.DataFrame(columns=['text', 'embedding'])
    return jsonify({'session_id': session_id}), 200

# 2. Route to add text for a specific session
@app.route('/add_text', methods=['POST'])
def add_text():
    global session_data
    data = request.json
    session_id = data.get('session_id')
    text = data.get('text')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Check if the session exists
    if session_id not in session_data:
        return jsonify({'error': f'No session found with ID {session_id}'}), 404
    
    session_df = session_data[session_id]
    
    # Add text to the session's DataFrame
    session_data[session_id] = add_text_to_dataframe(session_df, text)
    
    return jsonify({'message': 'Text added successfully', 'text': text}), 200

# 3. Route to search for relevant texts for a specific session
@app.route('/search', methods=['POST'])
def search():
    global session_data
    data = request.json
    session_id = data.get('session_id')
    query = data.get('query')
    top_n = data.get('top_n', 100)  # Default to 100 if not specified
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check if the session exists
    if session_id not in session_data or session_data[session_id].empty:
        return jsonify({'error': f'No data found for session_id {session_id}'}), 404
    
    session_df = session_data[session_id]
    
    # Perform the search for the session's DataFrame
    top_texts, top_scores = texts_ranked_by_relatedness(query, session_df, top_n)
    
    return jsonify({'top_texts': top_texts, 'top_scores': top_scores}), 200

@app.route('/get_session_data', methods=['GET'])
def get_session_data():
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    
    if session_id not in session_data:
        return jsonify({'error': 'No data found for session_id'}), 404
    
    session_df = session_data[session_id]
    
    return jsonify(session_df.to_dict(orient='records')), 200


# Run the Flask app
if __name__ == '__main__':
    app.run(host ="0.0.0.0", debug=True)
