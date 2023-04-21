# Author: Daniel Wrede
# Date: 2019-05-15

# Here the location is extracted from a message using spacys NER model.

import re
import spacy
nlp = spacy.load("de_core_news_lg")
from rake_nltk import Rake
r = Rake(language="german")


def spacy_ner(message):
    doc = nlp(message)
    place = []
    topic = []
    misc = []
    for ent in doc.ents:
        if ent.label_ == "LOC":
            place.append(ent.text)
        if ent.label_ == "ORG":
            topic.append(ent.text)
        if ent.label_ == "MISC":
            misc.append(ent.text)

    return place, topic, misc


def rake_keywords(message, max_length=3):
    r.extract_keywords_from_text(message)
    scored_keywords = r.get_ranked_phrases_with_scores()

    # Filter out keywords longer than the max_length
    filtered_keywords = [kw for score, kw in scored_keywords if len(kw.split()) <= max_length]

    # Find the original case of the keywords in the message
    original_case_keywords = []
    for keyword in filtered_keywords:
        # Escape any special characters in the keyword for regex search
        escaped_keyword = re.escape(keyword)
        # Search for the keyword in the message, case insensitive
        match = re.search(escaped_keyword, message, flags=re.IGNORECASE)
        if match:
            # Append the matched original keyword to the list
            original_case_keywords.append(match.group())

    return original_case_keywords


