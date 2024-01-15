import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract questions and answers
questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]

# Preprocess and vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vectorized_questions = vectorizer.fit_transform(questions)

def retrieve_answer(user_query):
    """
    Retrieve the most relevant answer based on the user's query.
    """
    # Vectorize the user query
    query_vector = vectorizer.transform([user_query])
    
    # Compute cosine similarity with predefined questions
    similarity_scores = cosine_similarity(query_vector, vectorized_questions).flatten()
    
    # Find the index of the highest similarity score
    best_match_index = similarity_scores.argmax()
    
    return answers[best_match_index]

if __name__ == "__main__":
    print("AECCGPT is still under construction but still I can help you! Type 'exit' or 'quit' to end our conversation.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = retrieve_answer(user_query)
        print(f"Bot: {response}")