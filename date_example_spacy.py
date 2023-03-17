import json, spacy
import re
#import IPython
from daterangeparser import parse

nlp = spacy.load("en_core_web_lg") #de_core_news_lg")

text = """How to ask which day of the week 
If you want to ask what day of the week it is, say:

What day is it today? or What’s the day today?

What day is it tomorrow? or What’s the day tomorrow?

 

To answer these questions you can say,

It’s Monday today. or Today is Monday.
1990
It’s Tuesday tomorrow. or Tomorrow is Tuesday.
23.12.1990
How to ask the date
If you want to ask what the date is, you can say:
23rd of December 1990
What’s the date today? or What’s today’s date?
7th of November 1992
What’s the date tomorrow? or What’s tomorrow’s date?

You can answer by saying:

It’s 27th September. / Today is 27th September.

Tomorrow is September 28th.

How to say the date
When we say dates in English we use ordinal numbers. So for 1 January, we don’t say the cardinal number ‘one’ but we say ‘first’. And we say ‘the’ before the number followed by ‘of’. For example,

It’s the first of January.

It’s also possible to invert the month and day. For example,

It’s January first. 

In this case you don’t need to say ‘the’ and ‘of’."""

def dep_subtree(token, dep):
  deps =[child.dep_ for child in token.children]
  child=next(filter(lambda c: c.dep_==dep, token.children), None)
  if child != None:
    return " ".join([c.text for c in child.subtree])
  else:
    return ""

# to remove citations, e.g. "[91]" as this makes problems with spaCy
p = re.compile(r'\[\d+\]')

def extract_events_spacy(line):
  line=p.sub('', line)
  events = []
  doc = nlp(line)
  for ent in filter(lambda e: e.label_=='DATE',doc.ents):
    try:
      start,end = parse(ent.text)
    except:
      # could not parse the dates, hence ignore it
      continue
    current = ent.root
    while current.dep_ != "ROOT":
      current = current.head
    desc = " ".join(filter(None,[
                                 dep_subtree(current,"nsubj"),
                                 dep_subtree(current,"nsubjpass"),
                                 dep_subtree(current,"auxpass"),
                                 dep_subtree(current,"amod"),
                                 dep_subtree(current,"det"),
                                 current.text, 
                                 dep_subtree(current,"acl"),
                                 dep_subtree(current,"dobj"),
                                 dep_subtree(current,"attr"),
                                 dep_subtree(current,"advmod")]))
    events = events + [(start,ent.text,desc)]
    return events
  
def main():
    # with open('telegram_messages.json', 'r') as f:
    #     # read the list from the file
    #     messages_dict = json.load(f)
    doc = nlp(text)
    for ent in filter(lambda e: e.label_=='DATE',doc.ents):
        print(ent.text)
    # for message in messages_dict:
        #print(message['message'])
        # Parse the message text with spacy
    # event = extract_events_spacy(text) #message['message'])
    # print(event)

        #doc = nlp(message['message'])
    


main()