import spacy
import requests

text_de = "Es ist der 1. Januar 2020. 2023-03-05 12:10:46+00:00. Heute Abend gibt es Ecstatic Dance in der Emil Thoma Schule der neuen Turnhalle. Ab 18 Uhr eine Stunde. 2023-02-28. 4 Tage ab Montag. bis 20.03.2023. Wir haben Zeit von Donnerstag den 18. (Himmelfahrt) bis Sonntag den 21. Mai. Datum : Freitag 10.03.2023 19:00 Ankommen 19:15 Anfangskreis, meditativer Einstieg 19:30 die erste Musikwelle beginnt. Ecstatic Dance Wochenende 18.-21. Mai. Samstag 18.03. 2023-03-15"
# read .txt file 'deutsche_Geschichte.txt' from directory
with open('deutsche_Geschichte.txt', 'r') as f:
    text_de = f.read()


response = requests.get('https://raw.githubusercontent.com/qualicen/timeline/master/history_of_germany.txt')
text_en = response.text
print('Loaded {} lines'.format(text_de.count('\n')))

# Load spacy model
nlp = spacy.load("de_core_news_lg") # de_core_news_lg en_core_web_lg

# Parse the message text with spacy
doc = nlp(text_de)

# Loop through the entities in the doc
counter = 0
for ent in filter(lambda e: e.label_=='DATE',doc.ents):
  print(ent.text)
  counter += 1
print("Found {} dates".format(counter))
# for ent in filter(lambda e: e.label_=='DATE',doc.ents):
#     print(ent.text)

# print("second loop")
# for ent in doc.ents:
#     # Print the entity text and label
#     print(ent.text, ent.label_)