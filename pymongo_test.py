from pymongo import MongoClient

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    # Attempt to retrieve server information
    server_info = client.server_info()
    print("Connected to MongoDB:", server_info)
except Exception as e:
    print("Failed to connect to MongoDB:", e)
