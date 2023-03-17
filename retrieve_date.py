import json, spacy


# Load spacy model
nlp = spacy.load("de_core_news_lg")

# open the file for reading
with open('telegram_messages.json', 'r') as f:
    # read the list from the file
    messages_dict = json.load(f)



# Loop through the messages
for message in messages_dict:
    #print(message['message'])
    # Parse the message text with spacy
    doc = nlp(message['message'])

    # for ent in doc.ents:
    #     print("{} -> {}".format(ent.text,ent.label_))

    # Loop through the entities in the doc
    for ent in filter(lambda e: e.label_=='DATE',doc.ents):
        print(ent.text)
        
    # Loop through the entities in the doc
    for ent in doc.ents:
        # Check if the entity is a date
        if ent.label_ in ["DATE"]: #, "GPE", "LOC", "PERSON", "ORG"]:
            # Print the entity text and label
            print(ent.text, ent.label_)
