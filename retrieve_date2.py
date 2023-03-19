import spacy
import requests
import stanza
import spacy_stanza
import datefinder # finds too many and wrong dates
from htmldate import find_date # only works for html files
# date parser only translate dates

language = 'de' # 'en'/'de'
text_de = "Es ist der 1. Januar 2020. 2023-03-05 12:10:46+00:00. Heute Abend gibt es Ecstatic Dance in der Emil Thoma Schule der neuen Turnhalle. Ab 18 Uhr eine Stunde. 2023-02-28. 4 Tage ab Montag. bis 20.03.2023. Wir haben Zeit von Donnerstag den 18. (Himmelfahrt) bis Sonntag den 21. Mai. Datum : Freitag 10.03.2023 19:00 Ankommen 19:15 Anfangskreis, meditativer Einstieg 19:30 die erste Musikwelle beginnt. Ecstatic Dance Wochenende 18.-21. Mai. Samstag 18.03. 2023-03-15"
text_de2 = "Es ist der 1. Januar 2020. - Gestern hat es geschneit. 21. Mai. heute." #Heute Abend Ab 18 Uhr eine Stunde. - 2023-02-28. - 4 Tage ab Montag. - Bis 20.03.2023. - Von Donnerstag den 18. bis zum 21. Mai. - Samstag 18.03. - 2023-03-15"

if language == 'de':
  # read .txt file 'deutsche_Geschichte.txt' from directory
  with open('deutsche_Geschichte.txt', 'r') as f:
    text = f.read()
  print("german")
  print(text[0:100])
  print(type(text))
elif language == 'en':
  response = requests.get('https://raw.githubusercontent.com/qualicen/timeline/master/history_of_germany.txt')
  text = response.text
  print("english")
else:
  print('Language not supported')
  exit()

# htmldate:
# date = find_date('https://htmldate.readthedocs.io/en/latest/index.html')
# print("htmldate: ", date)
# exit()

# datefinder:
# matches = datefinder.find_dates(text_de)
# counter = 0
# for match in matches:
#     print(match)
#     counter += 1
# print('Found {} dates'.format(counter))
# exit()

# Load spacy model
nlp = spacy.load("de_core_news_lg") # de_core_news_lg de_dep_news_trf 
#nlp = spacy.load("en_core_web_lg") # works best on english and german text

# Initialize a German pipeline with the `hdt` package
#nlp = spacy_stanza.load_pipeline("de", package="hdt")

# Parse the message text with spacy
print('Loaded {} lines'.format(text.count('\n')))
doc = nlp(text_de2)

# label_list = ["ADJA", "ADJD", "ADV", "APPO", "APPR", "APPRART", "APZR", "ART", "CARD", "FM", "ITJ", "KOKOM", "KON", "KOUI", "KOUS", "NE", "NN", "NNE", "PDAT", "PDS", "PIAT", "PIS", "PPER", "PPOSAT", "PPOSS", "PRELAT", "PRELS", "PRF", "PROAV", "PTKA", "PTKANT", "PTKNEG", "PTKVZ", "PTKZU", "PWAT", "PWAV", "PWS", "TRUNC", "VAFIN", "VAIMP", "VAINF", "VAPP", "VMFIN", "VMINF", "VMPP", "VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP", "XY", "_SP", "$,", "$.", "$("]
NER_list = ["LOC", "MISC", "ORG", "PER"]
for label in NER_list:
  print(label, spacy.explain(label))

exit()
for word in doc:
  print(word.text, word.tag_, spacy.explain(word.tag_))
exit()
  #print(ent.text, ent.start_char, ent.end_char, ent.label_)

# Loop through the entities in the doc
counter = 0
for ent in filter(lambda e: e.label_=='DATE',doc.ents):
  print(ent.text)
  counter += 1
print("Found {} dates".format(counter))