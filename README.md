#Endpoint 1: Simply add message to the database for RAG

curl -X POST http://localhost:8888/add_text \
  -H "Content-Type: application/json" \
  -d '{
        "session_id": "session123",
        "text": "This is a sample text to add to the session."
      }'


Sample Response:

{
  "message": "Text added successfully",
  "text": "This is a sample text to add to the session."
}


**Note:**
Please use the same session id for the same session.

#Endpoint 2: Seach using sessionId and query

curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -d '{
        "session_id": "session123",
        "query": "sample query to search",
        "top_n": 10
      }'

Sample Response:
  "top_scores": [
    0.7701478482119025,
    0.12600944507133105
  ],
  "top_texts": [
    "Amir is the best",
    "This is a sample text to add to the session."
  ]
}


Endpoint 3: Get all session messages

curl "http://localhost:8888/get_session_data?session_id=session123"

Sample Response:

[
  {
    "text": "This is a sample text to add to the session.",
    "timestamp": "Mon, 23 Sep 2024 10:14:06 GMT"
  },
  {
    "text": "Amir is the best",
    "timestamp": "Mon, 23 Sep 2024 10:17:04 GMT"
  }
]

Endpoint 4: Add to Long term memory using uuid
curl -X POST http://localhost:8888/add_to_long_term_memory \
  -H "Content-Type: application/json" \
  -d '{"uuid": "your_user_id", "text": "Your text here"}'

Sample Response:

{
  "message": "Text added to long-term memory",
  "text": "I am a student at the university of manchester"
}

Endpoint 5: Search Long term memory using uuid and query(prompt)

curl -X POST http://localhost:8888/search_long_term_memory \
  -H "Content-Type: application/json" \
  -d '{"uuid": "your_user_id", "query": "Your query here", "top_n": 5}'

Sample Response:

{
  "top_scores": [
    0.3520803910969268,
    0.1517654605923644
  ],
  "top_texts": [
    "I am a big fan of linkin park",
    "I am a student at the university of manchester"
  ]
}

Endpoint 6: Relative search which returns the relative index [x], where i is the current index of prompt and x and i-x is the index of the most relevant message:

Example 1: 

curl -X POST http://localhost:8888/resolve_references \
  -H "Content-Type: application/json" \
  -d '{"session_id": "111", "text": "Tell me more about it"}'

Sample Response:
{
  "relative_indices": [
    1
  ]
}

Example 2:

curl -X POST http://localhost:8888/resolve_references \
  -H "Content-Type: application/json" \
  -d '{"session_id": "111", "text": "Which one is better?"}'

Sample Response:

{
  "relative_indices": [
    1,
    2
  ]
}

Example 3:

curl -X POST http://localhost:8888/resolve_references \
  -H "Content-Type: application/json" \
  -d '{"session_id": "111", "text": "What was my last message?"}'

Sample Response:
{
  "relative_indices": [
    1
  ]
}


**to do**
1. Increase number of keywords and phrases in dictionaries
2. implement fuzzy search for mispelled words
3. Conditionally call resolve_references endpoint based on keywords





