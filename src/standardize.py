import re
from nltk.stem.snowball import SnowballStemmer
import nltk
from multiprocessing import Pool
from functools import partial


# Standardize
def remove_discord_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # other emoticons
                               u"\U000024C2-\U0001F251"  # emojis
                               "]+", flags=re.UNICODE)

    return emoji_pattern.sub(" ", text)


def remove_urls(text):
    return re.sub(r"https?://\S+", " ", text)


def remove_mentions(text):
    return re.sub(r"@\S+", "", text)


def remove_custom_emojis(text):
    txt = re.sub(r":\b(?!\d\d)\w+\b:", " ", text)
    return txt


def special_to_space(text):  # Excludes ' because we need it for contractions
    return re.sub(r"[^a-zA-Z0-9'\s]+", " ", text)


def compress_whitespace(text):
    return re.sub(r"\s+", " ", text)


def standardize_text(text, stemmer):
    import contractions

    txt = text.lower()

    txt = remove_urls(txt)
    txt = remove_mentions(txt)
    txt = remove_discord_emojis(txt)
    txt = remove_custom_emojis(txt)

    txt = special_to_space(txt)

    words = txt.split()
    txt = ""
    for word in words:
        word = contractions.fix(word)
        word = re.sub(r"\'", "", word)
        txt += f"{word} "  # space for separation

    words = txt.split()
    txt = ""
    for word in words:
        word = stemmer(word)
        txt += f"{word} "  # space for separation

    txt = compress_whitespace(txt)
    txt = txt.strip()

    return txt


def standardize_parallel(text_list, workers=12):
    nltk.download("punkt")
    stemmer = SnowballStemmer("english")

    standardizer_with_args = partial(standardize_text, stemmer=stemmer.stem)

    with Pool(processes=workers) as pool:
        output = pool.map(standardizer_with_args, text_list)

    return output
