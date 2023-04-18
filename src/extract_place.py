# Author: Daniel Wrede
# Date: 2019-05-15

# Here the location is extracted from a message using spacys NER model.

import spacy
nlp = spacy.load("de_core_news_lg")


def extract_place(message):
    doc = nlp(message)
    place = ""
    topic = ""
    misc = ""
    for ent in doc.ents:
        if ent.label_ == "LOC":
            place = place + ent.text + ", "
        if ent.label_ == "ORG":
            topic = topic + ent.text + ", "
        if ent.label_ == "MISC":
            misc = misc + ent.text + ", "

    return place, topic, misc