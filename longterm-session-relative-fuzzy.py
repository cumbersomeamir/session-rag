from flask import Flask, request, jsonify
import pandas as pd
import os
import openai
import uuid
import pymongo
from pymongo import MongoClient
from datetime import datetime
from thefuzz import fuzz  # Added for fuzzy matching

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
embeddings_collection = db['embeddings']  # Collection to store session-based embeddings
long_term_memory_collection = db['long_term_memory']  # Collection to store long-term memory embeddings

# Embedding creation function
def create_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Add a new text and its embedding to MongoDB (session-based)
def add_text_to_db(session_id, text):
    text_embedding = create_embedding(text)
    # Get the highest index in the session
    last_doc = embeddings_collection.find({'session_id': session_id}).sort('index', -1).limit(1)
    last_index = 0
    for doc in last_doc:
        last_index = doc.get('index', 0)
    index = last_index + 1
    document = {
        'session_id': session_id,
        'text': text,
        'embedding': text_embedding,
        'timestamp': datetime.utcnow(),
        'index': index
    }
    embeddings_collection.insert_one(document)
    return document

# Function to rank texts by relatedness (session-based)
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
    from scipy import spatial
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

# Function to decide whether to store text in long-term memory
def should_store_in_long_term_memory(prompt):
    completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": "You are a long term memory boolean program"},
        {"role": "user", "content": f"This is the prompt {prompt} Tell me if it is valuable information to store about the user or not Give either 0 or 1 as output no other text "}
      ]
    )
    response = completion.choices[0].message.content.strip()
    print("The response of the Boolean long term memory is ", response)
    return response

# Add a new text to long-term memory
def add_text_to_long_term_memory(uuid, text):
    text_embedding = create_embedding(text)
    document = {
        'uuid': uuid,
        'text': text,
        'embedding': text_embedding,
        'timestamp': datetime.utcnow()
    }
    long_term_memory_collection.insert_one(document)
    return document

# Function to rank texts by relatedness in long-term memory
def texts_ranked_by_relatedness_long_term(query, uuid, top_n=100):
    query_embedding = create_embedding(query)
    
    # Fetch all embeddings for the uuid
    cursor = long_term_memory_collection.find({'uuid': uuid})
    
    texts = []
    embeddings = []
    for doc in cursor:
        texts.append(doc['text'])
        embeddings.append(doc['embedding'])
    
    # Compute similarities
    from scipy import spatial
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
            'timestamp': doc.get('timestamp'),
            'index': doc.get('index')
        })
    
    if not data:
        return jsonify({'error': 'No data found for session_id'}), 404
    
    return jsonify(data), 200

# Route to add text to long-term memory, conditionally
@app.route('/add_to_long_term_memory', methods=['POST'])
def add_to_long_term_memory():
    data = request.json
    uuid = data.get('uuid')
    text = data.get('text')
    
    if not uuid:
        return jsonify({'error': 'User ID (uuid) is required'}), 400
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    # Decide whether to store the text
    store_decision = should_store_in_long_term_memory(text)
    
    if store_decision == '1':
        # Add text to long-term memory
        document = add_text_to_long_term_memory(uuid, text)
        return jsonify({'message': 'Text added to long-term memory', 'text': text}), 200
    else:
        return jsonify({'message': 'Text not added to long-term memory'}), 200

# Route to search long-term memory using uuid and query
@app.route('/search_long_term_memory', methods=['POST'])
def search_long_term_memory():
    data = request.json
    uuid = data.get('uuid')
    query = data.get('query')
    top_n = data.get('top_n', 100)  # Default to 100 if not specified
    
    if not uuid:
        return jsonify({'error': 'User ID (uuid) is required'}), 400
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    # Check if there is data for the uuid
    if long_term_memory_collection.count_documents({'uuid': uuid}) == 0:
        return jsonify({'error': f'No data found for user {uuid}'}), 404
    
    # Perform the search
    top_texts, top_scores = texts_ranked_by_relatedness_long_term(query, uuid, top_n)
    
    return jsonify({'top_texts': top_texts, 'top_scores': top_scores}), 200

# Updated Route to resolve references in the current session using heuristic approach
@app.route('/resolve_references', methods=['POST'])
def resolve_references():
    data = request.json
    session_id = data.get('session_id')
    text = data.get('text')  # Current message

    if not session_id:
        return jsonify({'error': 'Session ID is required'}), 400
    if not text:
        return jsonify({'error': 'Text is required'}), 400

    # Get the current index for the session
    last_doc = embeddings_collection.find({'session_id': session_id}).sort('index', -1).limit(1)
    last_index = 0
    for doc in last_doc:
        last_index = doc.get('index', 0)
    current_index = last_index + 1  # Index of the current message

    # Lowercase the current message
    text_lower = text.lower()

    # Heuristic rules to determine references
    relative_indices = []

    # Expanded patterns with additional keywords and phrases
    patterns = [
        (('it', 'this', 'that'), [1]),
        (('they', 'them', 'those', 'these'), [1, 2]),
        (('which one', 'which ones'), [1, 2]),
        (('what was my last message', 'what did i just say'), [1]),
        (('tell me more about', 'elaborate on'), [1]),
        (('earlier', 'previously', 'before', 'earlier you mentioned'), [1, 2, 3]),
        (('as mentioned', 'as i said', 'as you said', 'as previously noted'), [1, 2]),
        (('above', 'above message', 'prior message'), [1, 2]),
        (('in your last message', 'in your previous message', 'you previously said'), [1]),
        (('what did you say about', 'can you remind me about'), [1]),
        (('remind me about', 'refresh my memory on'), [1]),
        (('more details on', 'more information on', 'explain further about', 'go deeper into'), [1]),
        (('the previous topic', 'the earlier discussion', 'the last point'), [1, 2]),
        # Additional patterns to cover more cases
        (('continue with', 'continue about'), [1]),
        (('can you expand on', 'could you elaborate'), [1]),
        (('what do you mean by', 'clarify about'), [1]),
        (('i am confused about', 'i don\'t understand'), [1]),
    ]

    threshold = 80  # Threshold for fuzzy matching (0-100)

    for keywords, indices in patterns:
        for keyword in keywords:
            similarity = fuzz.partial_ratio(keyword, text_lower)
            if similarity >= threshold:
                relative_indices.extend(indices)
                break  # No need to check other keywords in this pattern

    # Remove duplicates and sort
    relative_indices = sorted(set(relative_indices))

    return jsonify({'relative_indices': relative_indices}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port='8888', debug=True)
