# Text-Based Question Answering System using TF-IDF and Cosine Similarity

This script is designed to implement a simple question-answering system. It utilizes the Term Frequency-Inverse Document Frequency (TF-IDF) method for vectorizing questions and employs cosine similarity to find the most relevant answer to a user's query from a predefined dataset.

## Code Explanation

### Importing Libraries

`import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity` 

-   `json`: Used for loading and parsing JSON data.
-   `sklearn.feature_extraction.text.TfidfVectorizer`: A tool for converting a collection of raw documents to a matrix of TF-IDF features.
-   `sklearn.metrics.pairwise.cosine_similarity`: Function for computing similarity as the normalized dot product of vectors.

### Loading Data from JSON File

`with open('data.json', 'r') as file:
    data = json.load(file)` 

-   This block of code opens a JSON file named `data.json` and loads its contents into the variable `data`.

### Extracting Questions and Answers

`questions = [item['question'] for item in data]
answers = [item['answer'] for item in data]` 

-   The script extracts lists of `questions` and corresponding `answers` from the loaded data.

### Preprocessing and Vectorizing Questions with TF-IDF

`vectorizer = TfidfVectorizer(stop_words='english')
vectorized_questions = vectorizer.fit_transform(questions)` 

-   `TfidfVectorizer` is initialized with English stop words filtered out.
-   The `questions` are transformed into a vectorized form using TF-IDF.

### Define the Function to Retrieve Answers

`def retrieve_answer(user_query):
    # Function details...` 

-   `retrieve_answer` is a function that takes a `user_query` as input and returns the most relevant answer from the dataset.

#### Functionality of `retrieve_answer`

 `query_vector = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vector, vectorized_questions).flatten()
    best_match_index = similarity_scores.argmax()
    return answers[best_match_index]` 

-   The user's query is vectorized using the same TF-IDF vectorizer.
-   The cosine similarity between the query vector and all question vectors is computed.
-   The index of the highest similarity score is found, and the corresponding answer is returned.

### Main Execution Block

`if __name__ == "__main__":
    # Interactive loop...` 

-   This part of the script provides an interactive loop where the user can input their queries.
-   The script keeps running until the user types "exit" or "quit".

#### Interactive Loop

 `while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = retrieve_answer(user_query)
        print(f"Bot: {response}")` 

-   The loop continually prompts the user for input.
-   If the input is "exit" or "quit", the script terminates.
-   Otherwise, it retrieves and prints the most relevant answer using the `retrieve_answer` function.
