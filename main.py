import streamlit as st
import pickle
import numpy as np
import re
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class BagofWords:
    def __init__(self):
        self.vocab = {}

    def fit(self, sentences):
        word_freq = {}
        for sentence in sentences:
            words = sentence.split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {word: idx for idx, (word, _) in enumerate(sorted_words)}

    def vectorize(self, sentence):
        vec = {}
        for word in sentence.lower().split():
            if word in self.vocab:
                idx = self.vocab[word]
                vec[idx] = vec.get(idx, 0) + 1
        return vec

import __main__
__main__.BagofWords = BagofWords


# =========================
# LOAD FILES
# =========================

model = pickle.load(open(os.path.join(BASE_DIR, "multinomial_naive_bayes_model.pkl"), "rb"))
bow = pickle.load(open(os.path.join(BASE_DIR, "bag_of_words_vectorizer.pkl"), "rb"))

# =========================
# PREPROCESS FUNCTION
# =========================
stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def preprocess_text(text):
    # Removes URLs (0.5 marks)
    text = re.sub(r"http\S+|www\S+", "", text)
    # Removes punctuation and non-alphanumeric characters (0.5 marks)
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    # Converts text to lowercase (0.5 marks)
    text = text.lower()
    # Removes extra whitespace (0.5 marks)
    text = re.sub(r"\s+", " ", text).strip()
    # Removes stopwords (2 marks)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)

# =========================
# CONVERT TO BAG OF WORDS
# =========================
def dicts_to_array(dict_list, vocab_size, dtype=np.uint8):
    arr = np.zeros((len(dict_list), vocab_size), dtype=dtype)
    for i, d in enumerate(dict_list):
        for idx, count in d.items():
            arr[i, idx] = min(count, 255)
    return arr
# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Text Classifier")

st.title("News Article Classification App")

st.write("Enter a news article in text form and find out which category the predicted class belongs to.")

# user input
user_input = st.text_area("Enter News Article")

# predict button
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter a news article.")
    else:

        # preprocessing
        processed_words = preprocess_text(user_input)
        bow_vector = bow.vectorize(processed_words)

        # convert to array
        final_input = dicts_to_array([bow_vector], len(bow.vocab), dtype=np.uint8)

        # prediction
        prediction = model.predict(final_input)

        # output
        class_mapping = {1: 'World', 2: 'Sports', 3: 'Business', 4: 'Sci/Tech'}
        predicted_category = class_mapping.get(prediction[0], 'Unknown')
        st.success(f"Predicted Class: {predicted_category}")
