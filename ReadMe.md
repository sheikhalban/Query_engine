# Text Query Engine
This code implements a `Text Query Engine` using natural language processing techniques to perform document retrieval based on user queries. The engine uses the TF-IDF (Term Frequency-Inverse Document Frequency) algorithm for text representation and cosine similarity for ranking the documents.


## Dependencies
The code requires the nltk library for natural language processing. It also uses the following modules from nltk:

- `word_tokenize` for tokenizing the text into words.
- `stopwords` for removing common English stopwords from the text.
- `WordNetLemmatizer` for word lemmatization.
Additionally, the code uses `sklearn` for vectorization and similarity calculations. It requires the following modules from `sklearn`:

-  `TfidfVectorizer` for transforming the text into TF-IDF vectors.
-  `cosine_similarity` for calculating cosine similarity between vectors.

Before running the code, make sure to download the necessary resources from the NLTK library by executing `nltk.download('wordnet')`.

## Usage
- Initialize an instance of the `TextQueryEngine` class.
- Build the index by providing a list of documents and their corresponding labels using the `build_index` method.
-  Perform a `query` by calling the query method with a query text.
- The engine will return a list of ranked documents based on their similarity to the query.

## Example
~~~
documents = [
    "I love to play soccer.",
    "I enjoy reading books.",
    "Programming is my passion.",
    "Football is an exciting sport.",
    "I fond of reading a book.",
    "I would like to go to see taj mahal",
    "I like to travel and explore new places."
]

labels = ["Sports", "Books", "Technology", "Sports", "Books", "Travel", "Travel"]

engine = TextQueryEngine()  # Create an instance of the TextQueryEngine class
engine.build_index(documents, labels)  # Build the index using the provided documents and labels

query_text = "travel"
results = engine.query(query_text)  # Perform a query using the query text

for label_documents in results:
    label = engine.labels[engine.documents.index(label_documents[0])]  # Get the label for the first document in the list
    print(f"Label: {label}")
    for document in label_documents:
        print(document)
    print()

~~~
## OUTPUT

~~~
Label: Travel
I like to travel and explore new places.
I would like to go to see taj mahal

Label: Books
I'm fond of reading a book.
I enjoy reading books.

Label: Technology
Programming is my passion.

Label: Sports
Football is an exciting sport.
I love to play soccer.
~~~

In the above example, the code creates a `TextQueryEngine` instance, builds an index using the provided documents and labels, and performs a query with the query text "travel". The results are then printed, which will contain the ranked documents related to the query text.

Feel free to modify the `documents` and `labels` to suit your specific use case and experiment with different query texts.
