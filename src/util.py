import pickle
from typing import List, Dict, Union
import os
import shutil
import re
import emoji


def load_pickled_data(path) -> Union[List[str], List[List[str]]]:
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def build_vocab(list_token_list: List[List[str]]) -> Dict:
    result_dict = {}
    for li in list_token_list:
        for token in li:
            if token not in result_dict:
                result_dict[token] = len(result_dict)
    return result_dict


def remove_file_or_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
            os.remove(path)
        except OSError as e:
            pass


def clean_personal_marker(phrase):
    """ Clean a clause extracted from a description"""

    if not phrase:
        return None

    # drop weird special characters
    phrase = phrase.encode('ascii', errors='ignore').decode().strip()
    x_prev = phrase

    while True:
        # remove excess whitespace
        phrase = re.sub(r"\s+", " ", phrase).strip()

        # remove personal markers such as I like X
        phrase = re.sub(r"^i (love|like|enjoy) ", "", phrase)
        # remove personal references such as I am a Y
        phrase = re.sub(r"^(i am|i'm|i'm) (a |an )?", "", phrase)
        # remove personal pronouns
        phrase = re.sub(r"^(i |a[n]?)\b", "", phrase)
        # remove unimportant words at the beginning and end of clause
        phrase = re.sub(r"^(and|the|from|to)\b", "", phrase)
        phrase = re.sub(r" of$", "", phrase)

        # removes social media links (snapchat, ig, email and phone address) mentions from bio
        phrase = re.sub(r'(on )?(snapchat|snap|ig|insta|instagram|email|phone): +[A-Za-z0-9_@.-]+', " ", phrase)

        # remove special characters
        phrase = re.sub(r'\u200d', "", phrase)

        # remove unimportant marks at the beginning and end of each phrase
        phrase = phrase.strip().strip(".,/!-]+[:)(-?'$%&_").strip()

        # remove some markers from the whole sentence
        phrase = re.sub(r"[!\(\)?.\{\}]", " ", phrase).strip()

        if phrase == x_prev:
            return phrase

        x_prev = phrase


def get_emoji_regexp():
    # Sort emoji by length to make sure multi-character emojis are
    # matched first
    emojis = sorted(emoji.EMOJI_DATA, key=len, reverse=True)
    pattern = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'
    return re.compile(pattern)


emojiexp = get_emoji_regexp()


def generate_personal_identifiers(description):
    """
    Splits up a profile description into a set of clauses. Returns the clauses and
    all emojis in the description (which are being treated as identity markers)
    """

    # lower cases the text
    # remove email addresses
    d = re.sub(r'\w+@\w+\.\w+', '', description.lower()).strip()
    # remove urls
    d = re.sub(r'http\S+', '', d).strip()
    # replace excess space characters
    d = d.replace("&emsp;", "").replace("&nbsp;", "")

    # get all emoji and treat them as split characters
    d = emojiexp.sub("|", string=d)  # .encode("ascii","namereplace").decode()

    # split on sensible split characters
    # | and
    spl = [x for x in re.split(
        r"[\(\)|•*;~°,\n\t]|[!…]+|[-–\/.]+ | [&+:]+ | [+] |([\/])(?=[A-Za-z ])|([.!-]{2,})| and |([#@][A-Za-z0-9_]+)",
        d.lower()) if (
                   x and x.strip() != "" and not x.strip() in "|•&*#;~°.!…-/–")]

    # clean all clauses
    spl = [clean_personal_marker(x) for x in spl]

    # remove weird things and things that become empty
    spl = [x for x in spl if x.strip() != "" and x.encode() != b'\xef\xb8\x8f']

    return spl


