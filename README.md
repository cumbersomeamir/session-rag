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




