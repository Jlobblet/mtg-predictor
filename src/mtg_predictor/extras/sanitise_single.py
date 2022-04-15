import re
from collections import Counter

from nltk import ClassifierI, PorterStemmer
from nltk.corpus import stopwords


def sanitise_single(text: str) -> Counter:
    remove = re.compile(r'[.,:;"()\[\]!?\-_]')
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    text = remove.sub("", text).lower()
    words = text.split()
    return Counter(
        stemmer.stem(word) for word in words if word and word not in stop_words
    )


def predict_single(model: ClassifierI, word_counts: dict):
    prediction = model.prob_classify(word_counts)
    print({s: f"{prediction.prob(s):.3f}" for s in prediction.samples()})
    return prediction


def predict(model: ClassifierI, text: str):
    word_counts = sanitise_single(text)
    return predict_single(model, word_counts)
