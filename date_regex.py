import re

text = "Es ist der 1. Januar 2020. 2023-03-05 12:10:46+00:00. Heute Abend gibt es Ecstatic Dance in der Emil Thoma Schule der neuen Turnhalle. Ab 18 Uhr eine Stunde. 2023-02-28. 4 Tage ab Montag. bis 20.03.2023. Wir haben Zeit von Donnerstag den 18. (Himmelfahrt) bis Sonntag den 21. Mai. Datum : Freitag 10.03.2023 19:00 Ankommen 19:15 Anfangskreis, meditativer Einstieg 19:30 die erste Musikwelle beginnt. Ecstatic Dance Wochenende 18.-21. Mai. Samstag 18.03. 2023-03-15"

# Define regular expression patterns for specific date formats
date_patterns = [
    r"\d{1,2}\.\s\w+\s\d{4}", # matches dates in the format "day month year", e.g. "1. Januar 2020"
    r"\d{4}-\d{2}-\d{2}", # matches dates in the format "YYYY-MM-DD", e.g. "2023-03-05"
    r"\d{1,2}\.\d{2}\.\d{4}", # matches dates in the format "DD.MM.YYYY", e.g. "10.03.2023"
    r"\d{2}\.\d{2}\.\d{4}", # matches dates in the format "DD.MM.YYYY", e.g. "18.03.2023"
    r"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}" # matches dates in the format "YYYY-MM-DD HH:MM:SS+hh:mm", e.g. "2023-03-05 12:10:46+00:00"
]

# Combine the patterns into a single regular expression
date_pattern = "|".join(date_patterns)

# Load the language model and process the text
nlp = spacy.load("de_core_news_lg")
doc = nlp(text)

# Use regular expressions to extract dates from the text
dates = []
for match in re.finditer(date_pattern, text):
    start, end = match.span()
    dates.append((start, end, match.group(0)))

# Print the extracted dates
for date in dates:
    print(date[2])
