import spacy
from spacy.matcher import Matcher

nlp = spacy.load("de_core_news_lg")

# Define the pattern for date matching
pattern = [{"LOWER": {"IN": ["montag", "dienstag", "mittwoch", "donnerstag", "freitag", "samstag", "sonntag"]}},
           {"IS_DIGIT": True, "LENGTH": 1},
           {"IS_PUNCT": True},
           {"IS_DIGIT": True, "LENGTH": 2},
           {"IS_PUNCT": True},
           {"IS_DIGIT": True, "LENGTH": 4}]

matcher = Matcher(nlp.vocab)
matcher.add("DATE", [pattern])

# Define the text to be matched
text = "Es ist der 1. Januar 2020. 2023-03-05 12:10:46+00:00. Heute Abend gibt es Ecstatic Dance in der Emil Thoma Schule der neuen Turnhalle. Ab 18 Uhr eine Stunde. 2023-02-28. 4 Tage ab Montag. bis 20.03.2023. Wir haben Zeit von Donnerstag den 18. (Himmelfahrt) bis Sonntag den 21. Mai. Datum : Freitag 10.03.2023 19:00 Ankommen 19:15 Anfangskreis, meditativer Einstieg 19:30 die erste Musikwelle beginnt. Ecstatic Dance Wochenende 18.-21. Mai. Samstag 18.03. 2023-03-15"

# Process the text and match the dates
doc = nlp(text)
matches = matcher(doc)

# Print the matched dates
for match_id, start, end in matches:
    date = doc[start:end].text
    print(date)
