import textstat
import nltk
import json
import string
import pyphen
import re


from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import words
from nltk.corpus import cmudict


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

dict_cmu = cmudict.dict()

pyphen.language_fallback('en_GB')
dic = pyphen.Pyphen(lang='en_GB')

with open("./utils/dict_awl.json") as file:
    dict_awl = json.load(file)
list_awl = [item for sublist in dict_awl.values() for item in sublist]
set_awl = set(list_awl)

with open("./utils/set_slur_words.json") as file:
    list_slur_words = json.load(file)
    set_slur_words = set(list_awl)

set_tags = set(['LS', 'TO', 'VBN', 'WP', 'UH', 'VBG', 'JJ', 'VBZ', 'VBP', 'NN', 'DT', 'PRP', 
                'WP$', 'NNPS', 'PRP$', 'WDT', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR',
                  'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS'])

english_words = set(words.words())

cached_stopwords = set(stopwords.words("english"))

def get_flesch_reading_ease(text: str):
    return textstat.flesch_reading_ease(text)

def get_gunning_fog(text: str):
    return textstat.gunning_fog(text)

def get_automated_readability_index(text: str):
    return textstat.automated_readability_index(text)

def get_smog_index(text: str):
    return textstat.smog_index(text)

def get_flesch_kincaid_grade(text: str):
    return textstat.flesch_kincaid_grade(text)

def get_coleman_liau_index(text: str):
    return textstat.coleman_liau_index(text)

def get_dale_chall_readability_score(text: str):
    return textstat.dale_chall_readability_score(text)

def get_automated_readability_index(text: str):
    return textstat.automated_readability_index(text)

def get_difficult_words(text: str):
    return textstat.difficult_words(text)

def get_linsear_write_formula(text: str):
    return textstat.linsear_write_formula(text)

def count_awl_words(tokens: list):
    return len([word for word in tokens if word in set_awl])

def calculate_lexical_diversity(text):
    words = text.split()
    unique_words = set(words)
    return len(unique_words) / len(words)

def get_pos_tags(text):
    tokens = word_tokenize(text)
    count_words = len(tokens)
    dict_pos_tags = {}
    pos_tags = nltk.pos_tag(tokens)
    res = [x[1] for x in pos_tags]
    for a in set_tags:
        dict_pos_tags.update({a: res.count(a)/count_words})
    dict_pos_tags = dict(sorted(dict_pos_tags.items()))
    return dict_pos_tags

def get_sentence_tree_roots(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    roots = [subtree.label() for subtree in tree.subtrees() if subtree.label() == 'NP']
    return roots

def get_sentence_tree_length(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def get_average_connections_at_root(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    root_connections = [len(subtree) for subtree in tree.subtrees() if subtree.label() == 'NP']
    return sum(root_connections) / len(root_connections) if root_connections else 0

def get_length_of_clauses(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    grammar = "CL: {<RB>?<VB.*><DT>?<JJ>*<NN>}"
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    clauses_lengths = [len(subtree) for subtree in tree.subtrees() if subtree.label() == 'CL']
    return sum(clauses_lengths) / len(clauses_lengths) if clauses_lengths else 0

def calculate_misspelling_score(text):
    tokens = word_tokenize(text.lower())
    # Extract tokens that are not in the English words set
    misspelled_tokens = [token for token in tokens if token not in english_words]
    # Calculate misspelling score as the ratio of misspelled tokens to total tokens
    misspelling_score = len(misspelled_tokens) / len(tokens) if len(tokens) > 0 else 0
    return misspelling_score

def detect_slur_usage(text):
    # You can customize this list with slurs or offensive words
    tokens = word_tokenize(text.lower())
    slur_count = sum(1 for token in tokens if token in set_slur_words)
    return slur_count

def calculate_overusage_of_punctuation(text):
    tokens = word_tokenize(text)
    # Extract punctuation tokens
    punctuation_tokens = [token for token in tokens if token in string.punctuation]
    # Calculate overusage of punctuation as the ratio of punctuation tokens to total tokens
    overusage_score = len(punctuation_tokens) / len(tokens) if len(tokens) > 0 else 0
    return overusage_score

def count_characters(text):
    text = text.replace(" ", "")
    return len(text)

def _nsyl(word):
  return [len(list(y for y in x if y[-1].isdigit())) for x in dict_cmu[word.lower()]]

def count_syllables(text):
    tokens = word_tokenize(text)
    nb_syl = 0
    for word in tokens:
        try:
            nb_syl += _nsyl(word)
        except:
            string = dic.inserted(word)
            string = string.split("-")
            nb_syl += len(string)
    return nb_syl

def count_words(text):
    tokens = word_tokenize(text)
    return len(tokens)

def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

def count_tagged_entity(text):
    tokens = word_tokenize(text)
    return len(re.findall(r'@(\w+)', text))/len(tokens)

def count_stop_words(text):
    tokens = word_tokenize(text)
    output = [word for word in tokens if word in cached_stopwords]
    return len(output)/len(tokens)

def count_quoted_words(text):
    tokens = word_tokenize(text)
    text = text.replace("'s", "")
    array = re.findall("(?<![\\w'])'([^']*)'(?!(?:'s|\\w))", text)
    array = [word_tokenize(x) for x in array]
    array = [item for sublist in array for item in sublist]
    array = [[char for char in token if char not in string.punctuation] for token in array]
    array = [item for item in array if len(item)>0]
    return len(array)/len(tokens)

def count_tabs(text):
    array = re.findall('(?<! ) {5,6}(?! )', text)
    return len(array)