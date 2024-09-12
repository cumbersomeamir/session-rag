# Importing the relevant libraries
import os
import openai
import pandas as pd  # For handling DataFrames
from scipy import spatial


from openai import OpenAI
client = OpenAI()

os.getenv("OPENAI_API_KEY")

texts = ["Amir Kidwai is a inventor and destroyer of worlds", "Ronaldo is the greatest of all time", "Abdulla Kidwai is a young boy", "Ayesha is a doctor"]  # Sample texts

'''1. Function to create embeddings'''
def create_embedding(text):

    response = client.embeddings.create(
        input= text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding

'''2. Create dataframe with embeddings'''
# Function to combine embeddings into a DataFrame
def create_dataframe_with_embeddings(texts):
    data = []
    
    for text in texts:
        embedding = create_embedding(text)
        data.append({'text': text, 'embedding': embedding})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


'''3. Define the search function'''

def strings_ranked_by_relatedness(query, df, top_n: int = 100) -> tuple[list[str], list[float]]:
    """Returns the top N most related strings and their relatedness scores."""
    
    # Get the embedding for the query string
    query_embedding = create_embedding(query)
    
    # Calculate relatedness between query and each string in the dataframe
    relatedness_scores = df['embedding'].apply(lambda emb: 1 - spatial.distance.cosine(query_embedding, emb))
    
    # Sort strings by relatedness scores in descending order
    sorted_df = df.assign(relatedness=relatedness_scores).sort_values(by='relatedness', ascending=False)
    
    # Return the top N strings and their relatedness scores
    top_strings = sorted_df['text'].head(top_n).tolist()
    top_scores = sorted_df['relatedness'].head(top_n).tolist()
    
    return top_strings, top_scores


# Creating the DataFrame
df = create_dataframe_with_embeddings(texts)

top_strings, top_scores = strings_ranked_by_relatedness("Who is Amir", df )

print("The top_strings are ", top_strings)
print("The top_scores are ", top_scores)
