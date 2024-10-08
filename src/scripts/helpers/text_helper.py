import re
from bs4 import BeautifulSoup
import contractions
import re
import nltk
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer


# Ensure that the words corpus is downloaded
nltk.download('words')
# Download the wordnet corpus
nltk.download('wordnet')
english_words = set(words.words())  # Load this once
# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

def is_english(text):
    # Remove punctuations using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    text_words = text.split()
    # Consider only the first 20 words
    first_20_words = text_words[:20]
    
    english_count = 0
    for word in first_20_words:
        # Lemmatize the word before checking
        lemmatized_word = lemmatizer.lemmatize(word.lower())
        if lemmatized_word in english_words:
            english_count += 1
    
    # return ration of english to non english words
    return english_count / len(first_20_words)

def normalize_text(text):
    """
    Normalize the text by stripping leading/trailing whitespace,
    converting to lowercase, and replacing multiple spaces with a single space.

    Parameters:
    text (str): The input text to normalize.

    Returns:
    str: The normalized text.
    """
    # Strip leading/trailing whitespace
    text = text.strip()
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

# not used
def remove_html_tags(text):
    """
    Remove HTML tags from the given text.

    Parameters:
    text (str): The input text containing HTML content.

    Returns:
    str: The cleaned text without HTML tags.
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# not used
def expand_contractions(text):
    """
    Expand contractions in the text (e.g., "don't" to "do not").

    Parameters:
    text (str): The input text with contractions.

    Returns:
    str: The text with expanded contractions.
    """
    return contractions.fix(text)

def clean_text(text):
    """
    Clean the text by performing the following operations:
    1. Replace occurrences of '#sep#' with a single space.
    2. Remove special characters (except for punctuation marks: .,!?;:).
    3. Remove all numbers.
    4. Replace multiple spaces with a single space.
    5. Return an empty string if the cleaned text contains no letters.

    Parameters:
    text (str): The input text to clean.

    Returns:
    str: The cleaned text or an empty string if no letters are present.
    """
    # Replace #sep# with a space
    text = text.replace('#sep#', ' ')
    
    # Strip leading/trailing whitespace and remove special characters and numbers, keeping punctuation
    text = re.sub(r'[^\w\s.,!?;:]|\d', '', text.strip())
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Check if the cleaned text contains any letters
    if not re.search(r'[a-zA-Z]', text):
        return ''
    
    return text