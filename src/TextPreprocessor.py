import string
import nltk
from string import digits, punctuation
from nltk.corpus import stopwords

from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import sent_tokenize
from nltk import pos_tag



def tokenize(document, strips=[''], lower=True, remove_stopwords=True, lemmatization=True, stemming=True):
    # Break the document into sentences
    for sent in sent_tokenize(document):
        # Break the sentence into part of speech tagged tokens
        for token, tag in pos_tag(wordpunct_tokenize(sent)):

            # get rid of the number in token
            remove_digits = str.maketrans('', '', digits)
            token = token.translate(remove_digits)

            # get rid of punctuation in token
            remove_punc = str.maketrans('', '', punctuation)
            token = token.translate(remove_punc)

            # turn all tokens to lower case
            token = token.lower() if lower else token

            if remove_stopwords:
                # get a list of english stop words
                stop_words = list(stopwords.words('english'))
                strips = strips + stop_words

            # ignore token in the strips
            if token in strips:
                continue

            # Lemmatize the token
            if lemmatization:
                lemmatizer = WordNetLemmatizer()
                token = lemmatizer.lemmatize(token)

            # Stem the token
            if stemming:
                ps = PorterStemmer()
                token = ps.stem(token)

            yield token


def my_tokenizer(doc, strips=[''], lower=True, remove_stopwords=True, lemmatization=True, stemming=True):
    """Customized tokenizer: remove number, punctuations and english stop words, use lemmatizing """
    return [token for token in tokenize(doc, strips, lower, remove_stopwords, lemmatization, stemming)]

