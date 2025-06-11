import numpy as np

import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('punkt_tab')

import warnings

warnings.filterwarnings("ignore")

import dill
from tensorflow.keras.models import load_model
from gensim.models import KeyedVectors

contractions = {
    # Common verb contractions
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",

    # Pronoun contractions
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that'll": "that will",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there'll": "there will",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "why'd": "why did",
    "why's": "why is",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",

    # Special cases and colloquialisms
    "'cause": "because",
    "let's": "let us",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "gov't": "government",
    "how'd'y": "how do you"
}


def pre_process(q):
    q = str(q).lower().strip()

    # replace the certail characters with their string replacements
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # the pattern [math] appears in the dataset around 900 so we remove it

    q = q.replace("[math]", " ")

    # replacing numbers with strings
    q = q.replace(',000,000,000', 'b')
    q = q.replace(',000,000', 'm')
    q = q.replace(',000', 'k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # removing the short form words

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # removing html tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q


def fetch_normal_features(row):
    q1 = row["question1"]
    q2 = row["question2"]

    normal_features = [0.0] * 7

    normal_features[0] = len(q1)
    normal_features[1] = len(q2)

    normal_features[2] = len(q1.split(" "))
    normal_features[3] = len(q2.split(" "))

    common_words = set(word for word in q1.split(" ") if word in q2.split(" "))
    normal_features[4] = len(common_words)

    normal_features[5] = len(set(q1.split(" "))) + len(set(q2.split(" ")))

    normal_features[6] = round(len(common_words) / normal_features[5], 2)

    return normal_features


def fetch_token_features(row):
    q1 = row["question1"]
    q2 = row["question2"]

    beta = 0.001

    STOP_WORDS = stopwords.words("english")

    token_features = [0.0] * 8

    # converting sentences into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # get the non stop words
    q1_words = set([words for words in q1_tokens if words not in STOP_WORDS])
    q2_words = set([words for words in q2_tokens if words not in STOP_WORDS])

    # GET THE STOP WORDS
    q1_stopwords = set([words for words in q1_tokens if words in STOP_WORDS])
    q2_stopwords = set([words for words in q2_tokens if words in STOP_WORDS])

    # GET THE COMMON NON STOPWORDS
    common_words_count = len(q1_words.intersection(q2_words))

    # get the common stopwords
    common_stopwords_count = len(q1_stopwords.intersection(q2_stopwords))

    # get the common tokens
    common_token_count = len(set(q1_tokens).intersection((q2_tokens)))

    token_features[0] = common_words_count / (min(len(q1_words), len(q2_words)) + beta)
    token_features[1] = common_words_count / (max(len(q1_words), len(q2_words)) + beta)
    token_features[2] = common_stopwords_count / (max(len(q1_stopwords), len(q2_stopwords)) + beta)
    token_features[3] = common_stopwords_count / (max(len(q1_stopwords), len(q2_stopwords)) + beta)
    token_features[4] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + beta)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + beta)

    # last words of the batch
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # first word of the batch
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def fetch_length_features(row):
    q1 = row["question1"]
    q2 = row["question2"]

    length_features = [0.0] * 3

    # converting sentences into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # absolute len feature
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # average token length
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    # longest substring ratio
    strs = list(distance.lcsubstrings(q1, q2))
    # Check if strs is not empty before accessing the first element
    if len(strs) > 0:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    else:
        # If no common substring is found, the ratio is 0
        length_features[2] = 0.0

    return length_features


def fetch_fuzzy_features(row):
    q1 = row["question1"]
    q2 = row["question2"]

    fuzzy_features = [0.0] * 4

    # fuzz ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz partial ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token sort ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token set ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features


def preprocess(text):
    return word_tokenize(text.lower())


def get_sentence_vector(sentence, model):
    words = preprocess(sentence)
    vectors = []
    for word in words:
        if word in model.key_to_index:
            vectors.append(model[word])
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Load feature names
feature_names = [
    "q1_len", "q2_len", "q1_words", "q2_words", "words_common", "words_total",
    "words_share", "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min",
    "ctc_max", "last_word_eq", "first_word_eq", "abs_len_diff", "mean_len",
    "longest_substring_ratio", "fuzz_ratio", "fuzz_partial_ratio",
    "fuzz_sort_ratio", "fuzz_set_ratio"
]

word_vectors = KeyedVectors.load('word2vec_vectors.kv', mmap='r')

# Load scaler
with open('feature_scaler.pkl', 'rb') as f:
    scaler = dill.load(f)

# Load model
model = load_model('duplicate_questions_model.h5')


def duplicate_prediction(q1, q2):
    # 1. Preprocess
    q1_processed = pre_process(q1)
    q2_processed = pre_process(q2)

    # 2. Feature engineering
    normal_features = fetch_normal_features({'question1': q1_processed, 'question2': q2_processed})
    token_features = fetch_token_features({'question1': q1_processed, 'question2': q2_processed})
    length_features = fetch_length_features({'question1': q1_processed, 'question2': q2_processed})
    fuzzy_features = fetch_fuzzy_features({'question1': q1_processed, 'question2': q2_processed})

    # 3. Combine features
    all_features = np.array(normal_features + token_features + length_features + fuzzy_features).reshape(1, -1)
    scaled_features = scaler.transform(all_features)

    # 4. Get embeddings
    q1_embedding = get_sentence_vector(q1_processed, word_vectors)
    q2_embedding = get_sentence_vector(q2_processed, word_vectors)

    # 5. Predict
    prediction = model.predict([
        q1_embedding.reshape(1, -1),
        q2_embedding.reshape(1, -1),
        scaled_features
    ])

    return prediction

