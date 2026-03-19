import re
import nltk
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not",
    "it's": "it is", "that's": "that is", "i'm": "i am",
    "they're": "they are", "we're": "we are", "you're": "you are"
}

FINANCIAL_ABBREV = {
    "eps": "earnings per share",
    "yoy": "year over year",
    "qoq": "quarter over quarter",
    "bps": "basis points",
    "ipo": "initial public offering",
    "fed": "federal reserve",
    "gdp": "gross domestic product",
    "etf": "exchange traded fund",
    "pe":  "price to earnings",
    "roi": "return on investment",
}

# these carry signal — never remove them
FINANCIAL_KEEP = {
    "not", "up", "down", "above", "below", "over",
    "under", "more", "less", "high", "low", "higher", "lower"
}


class FinancialCleaner:

    def __init__(self, expand_abbrev=True, remove_stopwords=False):
        self.expand_abbrev    = expand_abbrev
        self.remove_stopwords = remove_stopwords

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def remove_noise(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+",       "", text)
        text = re.sub(r"@\w+",                 "", text)
        text = re.sub(r"#\w+",                 "", text)
        text = re.sub(r"\$[a-z]{1,5}",         "", text)
        text = re.sub(r"\d+\.?\d*\s*bps",      " NUM bps ", text)
        text = re.sub(r"\d+\.?\d*%",           " PCT ", text)
        text = re.sub(r"\$\d+[\.,]?\d*[BMK]?", " NUM ", text)
        text = re.sub(r"\d+\.?\d*[BMK]",       " NUM ", text)
        text = re.sub(r"[^\w\s]",              " ", text)
        text = re.sub(r"\s+",                  " ", text).strip()
        return text

    def expand_financial_abbrev(self, text: str) -> str:
        if not self.expand_abbrev:
            return text
        tokens = text.split()
        tokens = [FINANCIAL_ABBREV.get(t, t) for t in tokens]
        return " ".join(tokens)

    def remove_stopwords_fn(self, text: str) -> str:
        tokens = text.split()
        tokens = [
            t for t in tokens
            if t not in STOP_WORDS or t in FINANCIAL_KEEP
        ]
        return " ".join(tokens)

    def clean(self, text: str) -> str:
        text = self.expand_contractions(text)
        text = self.remove_noise(text)
        text = self.expand_financial_abbrev(text)
        if self.remove_stopwords:
            text = self.remove_stopwords_fn(text)
        return text.strip()


def clean_batch(texts: list, **kwargs) -> list:
    cleaner = FinancialCleaner(**kwargs)
    return [cleaner.clean(t) for t in texts]