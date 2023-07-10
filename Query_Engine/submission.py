import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextQueryEngine:
    def __init__(self):
        self.documents = []  # List to store the original documents
        self.labels = []  # List to store the corresponding labels for the documents
        self.vectorizer = TfidfVectorizer()  # TF-IDF vectorizer for text representation
        self.stopwords = set(stopwords.words('english'))  # Set of stopwords for text preprocessing
        self.lemmatizer = WordNetLemmatizer()  # Lemmatizer for word lemmatization

    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())  # Tokenize the text into words and convert to lowercase
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]  # Lemmatize each word if it is alphanumeric
        tokens = [token for token in tokens if token not in self.stopwords]  # Remove stopwords from the tokens
        return ' '.join(tokens)  # Return the preprocessed text as a string

    def build_index(self, documents, labels):
        self.documents = documents  # Store the original documents
        self.labels = labels  # Store the corresponding labels
        preprocessed_documents = [self.preprocess_text(doc) for doc in documents]  # Preprocess each document
        self.vectorizer.fit_transform(preprocessed_documents)  # Fit the vectorizer on the preprocessed documents

    def query(self, query_text):
        preprocessed_query = self.preprocess_text(query_text)  # Preprocess the query text
        query_vector = self.vectorizer.transform([preprocessed_query])  # Transform the preprocessed query text into a vector
        similarities = cosine_similarity(query_vector, self.vectorizer.transform(self.documents))  # Calculate cosine similarity between the query vector and document vectors
        ranked_indices = similarities.argsort()[0][::-1]  # Sort indices in descending order of similarity
        ranked_documents = [self.documents[index] for index in ranked_indices]  # Retrieve the ranked documents based on the sorted indices
        ranked_labels = [self.labels[index] for index in ranked_indices]  # Retrieve the corresponding labels for the ranked documents
        results = []  # List to store the results
        unique_labels = set(ranked_labels)  # Get unique labels
        for label in unique_labels:
            label_indices = [i for i, l in enumerate(ranked_labels) if l == label]  # Find indices of documents with the current label
            label_documents = [ranked_documents[i] for i in label_indices]  # Get documents corresponding to the label
            results.append(label_documents)  # Append the documents to the results list
        return results  # Return the results
