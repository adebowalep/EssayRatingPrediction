# Librairies

from nltk.corpus import stopwords
from nltk import RegexpTokenizer, pos_tag
from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict
import nltk
import json
import pyphen
import spacy
import re

pyphen.language_fallback('en_GB')
dic = pyphen.Pyphen(lang='en_GB')

dict_cmu = cmudict.dict()

with open("./utils/dict_awl.json") as file:
    dict_awl = json.load(file)
list_awl = [item for sublist in dict_awl.values() for item in sublist]


nlp = spacy.load("en_core_web_trf")


with open("./utils/dale-chall-word-list.txt", "r") as file:
    array_dale_chall = file.read().splitlines()

array_dale_chall =[word.replace(" ", "") for word in array_dale_chall]
set_dale_chall = set(array_dale_chall)

# Preprocessing

def tokenize_essay(essay):
    return RegexpTokenizer(r"[a-zA-Z0-9]+").tokenize(str.lower(essay))

def lemmatize_essay(essay):
    text = nlp(essay)
    return [token.lemma_ for token in text]


def remove_stop_words(tokens):
    output = []
    for word in tokens:
        if word not in stopwords.words("english"):
            output.append(word)
    return output

def remove_punctuation(essay):
    essay = essay.replace("/", " ")
    essay = essay.replace("(", " ")
    essay = essay.replace(")", " ")
    essay = essay.replace("  ", " ")
    return re.sub(r'[^\w\s]', '', essay)

def remove_special_words(essay):
    return re.sub(r'@\w+', '', essay)

# Words and word structure

## Flesch reading ease --> numbers of words, numbers of sentences, numbers of syllables
## https://en.wikipedia.org/wiki/Flesch–Kincaid_readability_tests
## For Flesch kincaid score, we don't remove stopwords

def count_words(tokens):
    return(len(tokens))

def count_sentences(essay):
    return len(sent_tokenize(essay))

def _nsyl(word):
  return [len(list(y for y in x if y[-1].isdigit())) for x in dict_cmu[word.lower()]]

def count_syllables(tokens):
    nb_syl = 0
    for word in tokens:
        try:
            nb_syl += _nsyl(word)
        except:
            string = dic.inserted(word)
            string = string.split("-")
            nb_syl += len(string)
    return nb_syl

## Gunning fog --> complex words ("complex" words consisting of three or more syllables. Do not include proper nouns, familiar jargon, or compound words. Do not include common suffixes (such as -es, -ed, or -ing) as a syllable;)
## https://en.wikipedia.org/wiki/Gunning_fog_index

def count_complex_words(tokens):
    nb_complex_words = 0
    for word in tokens:
        if count_syllables(word) >= 3:
            nb_complex_words += 1
    return nb_complex_words



## Automated Readability Index --> number of characters
## https://en.wikipedia.org/wiki/Automated_readability_index

def count_characters(essay):
    essay = essay.replace(" ", "")
    return len(essay)

## SMOG --> Polysyllables (= words of 3 or more syllables, maybe similar to Gunning fog's "complex words") 
## https://en.wikipedia.org/wiki/SMOG

## Flesch kincaid : same factors than Flesch reading ease

## Coleman-Liau index --> average number of letters per 100 words, and average number of sentences per 100 words (maybe irrelevant since we already have numbers of letters and sentences)
## https://en.wikipedia.org/wiki/Coleman–Liau_index

## Dale-Chall readability formula --> list of difficult words (set of 3 000  familiar words)

def count_dale_chall_difficult_words(tokens):
    difficult_words = [word for word in tokens if word not in set_dale_chall]
    return len(difficult_words)


# Lexically

## Words sophistication thanks to a corpus (AWL)

def count_awl_words(tokens):
    return len([word for word in tokens if word in list_awl])

## Words frequency (tf-idf)
## I'm not sure we should use this measure, beaucause it seems field related

## Lexical diversity

def count_unique_lemme(tokens):
    return len(set(tokens))

## Lexical variation
set_tags = set(['LS', 'TO', 'VBN', 'WP', 'UH', 'VBG', 'JJ', 'VBZ', 'VBP', 'NN', 'DT', 'PRP', 
                'WP$', 'NNPS', 'PRP$', 'WDT', 'RB', 'RBR', 'RBS', 'VBD', 'IN', 'FW', 'RP', 'JJR',
                  'JJS', 'PDT', 'MD', 'VB', 'WRB', 'NNP', 'EX', 'NNS', 'SYM', 'CC', 'CD', 'POS'])

def get_pos_tags(tokens):
    dict_pos_tags = {}
    pos_tags = nltk.pos_tag(tokens)
    res = [x[1] for x in pos_tags]
    for a in set_tags:
        dict_pos_tags.update({a: res.count(a)})
    dict_pos_tags = dict(sorted(dict_pos_tags.items()))
    return dict_pos_tags


# Syntactic structure

## Roots of Sentence tree

## Lenght of Sentence tree

## Average number of Connections of Sentence tree at the root level

## Lenght of clauses

# Quality

## Mispelling score

# Use my previous research to find incorrect words

## Slur usage

# Maybe a lack of data (hard to find wordlist of slur)

## Overusage of punctuation

# Count the number of punctuation

