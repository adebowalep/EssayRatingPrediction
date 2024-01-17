import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import string
import numpy as np
from gensim.models import Word2Vec
import gensim.downloader as api




def analyze_essay_text(text):
    """
    Analyzes a given essay text and extracts various linguistic and statistical features.
    
    This function performs multiple analyses on the provided text, including counting 
    characters, words, capital characters, capital words, sentences, unique words, stopwords, 
    and punctuation marks. It also calculates the average word length, average sentence length, 
    the ratio of unique words to total words, and the ratio of stopwords to total words.
    
    The function is designed to be used on individual essays and is particularly useful for 
    natural language processing tasks, where understanding textual complexity and composition 
    is important.
    
    Parameters:
    - text (str): The essay text to be analyzed.
    
    Returns:
    - dict: A dictionary containing the following keys and their corresponding calculated values:
        - "Number of characters": The total number of characters in the text.
        - "Number of words": The total number of words in the text.
        - "Number of capital characters": The count of uppercase characters in the text.
        - "Number of capital words": The count of words in uppercase in the text.
        - "Number of sentences": The total number of sentences in the text.
        - "Number of unique words": The number of unique words in the text.
        - "Number of stopwords": The count of stopwords in the text, based on NLTK's English stopwords list.
        - "Number of punctuation": The count of punctuation marks in the text.
        - "Average word length": The average length of words in the text.
        - "Average sentence length": The average length of sentences in the text.
        - "Unique words vs word count feature": The ratio of unique words to total words in the text.
        - "Stopwords count vs words count feature": The ratio of stopwords to total words in the text.
    """
    # Count the number of characters
    num_characters = len(text)
    
    # Count the number of words
    words = word_tokenize(text)
    num_words = len(words)
    
    # Count the number of capital characters
    num_capital_characters = sum(1 for char in text if char.isupper())
    
    # Count the number of capital words
    num_capital_words = sum(1 for word in words if word.isupper())
    
    # Count the number of sentences
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    
    # Count the number of unique words
    unique_words = set(words)
    num_unique_words = len(unique_words)
    
    # Count the number of stopwords
    stop_words = set(stopwords.words('english'))
    num_stopwords = sum(1 for word in words if word.lower() in stop_words)
    
    # Count the number of punctuation
    num_punctuation = sum(1 for char in text if char in string.punctuation)
    
    # Calculate average word length
    total_word_length = sum(len(word) for word in words)
    average_word_length = total_word_length / num_words
    
    # Calculate average sentence length
    total_sentence_length = sum(len(sent) for sent in sentences)
    average_sentence_length = total_sentence_length / num_sentences
    
    # Calculate unique words vs word count feature
    unique_words_ratio = num_unique_words / num_words
    
    # Calculate stopwords count vs words count feature
    stopwords_ratio = num_stopwords / num_words
    
    # Return a dictionary of features
    return {
        "Number of characters": num_characters,
        "Number of words": num_words,
        "Number of capital characters": num_capital_characters,
        "Number of capital words": num_capital_words,
        "Number of sentences": num_sentences,
        "Number of unique words": num_unique_words,
        "Number of stopwords": num_stopwords,
        "Number of punctuation": num_punctuation,
        "Average word length": average_word_length,
        "Average sentence length": average_sentence_length,
        "Unique words vs word count feature": unique_words_ratio,
        "Stopwords count vs words count feature": stopwords_ratio
    }


def count_text_punctuation(text):
    """
    Counts the occurrences of each punctuation mark in the given text.

    This function iterates through each character in the text and tallies 
    the number of times each punctuation mark appears. It's useful for 
    text analysis tasks where understanding the frequency of punctuation 
    can give insights into writing style or text structure.

    Parameters:
    - text (str): The text in which to count punctuation marks.
    
    Returns:
    - dict: A dictionary where keys are punctuation marks and values 
            are the counts of each punctuation mark in the text.

    Example:
    >>> count_text_punctuation("Hello, world! How's it going?")
    {',': 1, '!': 1, '?': 1, "'": 1}
    """
    punctuation_counts = {}
    for char in text:
        if char in string.punctuation:
            punctuation_counts[char] = punctuation_counts.get(char, 0) + 1
    return punctuation_counts



def extract_tagged_entities(text):
    """
    Extracts tagged entities from the given essay text.

    This function identifies and extracts occurrences of specific tagged entities in the essay text. 
    These entities are marked with an '@' symbol followed by an alphanumeric identifier (e.g., @CAPS, 
    @NUM). Such tagging is often used in educational datasets to anonymize or standardize certain parts 
    of the text. Identifying these entities can be useful for various text analysis tasks within the 
    educational domain.

    Parameters:
    - text (str): The essay text from which to extract tagged entities.
    
    Returns:
    - list: A list of extracted tagged entities (without the '@' symbol).

    Example:
    >>> extract_tagged_entities("Dear @CAPS1, I scored @NUM1 in my test.")
    ['CAPS1', 'NUM1']
    """
    mentions = re.findall(r'@(\w+)', text)
    return mentions


def initialize_nltk_resources():
    """
    Function to download necessary NLTK resources.
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def text_preprocessing_pipeline(text):
    """
    Function to preprocess text data.
    """
    # Remove special tokens
    text = re.sub(r'@\w+', '', text)
    
    # Lowercasing
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Correct spelling errors
    #spell = SpellChecker()
    #tokens = [spell.correction(word) for word in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Handling numbers (depends on the requirement)
    tokens = ['NUM' if word.isdigit() else word for word in tokens]

    # Join tokens back to string
    text = ' '.join(tokens)

    return text

def get_average_vector(text, vector_size):
    """
    Function to calculate the average vector of a given essay.
    """
    # tokens
    tokens = text_preprocessing_pipeline(text)

    # Define and train the Word2Vec model
    model = Word2Vec(tokens, vector_size=100, window=5, min_count=1, workers=4)
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return [0] * vector_size  # Return a vector of zeros if no valid tokens
    average_vector = sum(vectors) / len(vectors)
    return average_vector

# Apply the function to get the average vector for each essay
#training_data['essay_vector'] = training_data['essay'].apply(lambda tokens: get_average_vector(tokens, model, 100))