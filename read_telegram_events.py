# Import libraries
import asyncio, configparser
from telethon import TelegramClient, utils
from telethon.tl import functions, types
from telethon.tl.functions.messages import GetFullChatRequest
import spacy
import logging

#logging.getLogger("asyncio").setLevel(logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
# Load spacy model for English
nlp = spacy.load("de_core_news_sm")

# Reading Configs
config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = str(config['Telegram']['api_hash'])
phone = config['Telegram']['phone']
username = config['Telegram']['username']

# Create a client object with your credentials
client = TelegramClient(username, api_id, api_hash)


# Define an async function to get messages from channel
async def get_messages():
    # Start the client
    await client.start()
    me = await client.get_me()
    print(f"{me.first_name} {me.last_name}")
    print("id ", me.id)
    print("phone ", me.phone)

    # async for dialog in client.iter_dialogs():
    #     if 'leandro'.lower() in dialog.name.lower():
    #         print(dialog.name, 'has ID', dialog.id)

    print("getting the channel entity by username")
    # Leandro 667560725
    # Philipp 300838665
    # 667560725
    # Conscious Freiburg -1001794946398
    chat_id = -1001794946398
    group_entity = await client.get_input_entity(chat_id) #300838665) #1794946398) #1236978515
    print("channel: ", group_entity.stringify())
    connection_status = client.is_connected()
    print("connection_status: ", connection_status)
    # chat = await client(functions.messages.GetFullChatRequest(chat_id))
    # print(chat)
    # Get the last messages from the channel
    # async for message in client.iter_messages(channel, limit=5):
    #     print(message.text)
    messages = await client.get_messages(group_entity, limit = 10) #limit=
    
    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if dialog.id == chat_id:
            print(f"{dialog.name}, {dialog.id}") #, {dialog.entity.phone}")


    # Loop through the messages
    for counter,message in enumerate(messages):
        # Check if the message is text
        print("message number: ", counter)

        if message.text:
            # Print the message text
            #print(message.text)
            # try: print(message.media.photo)
            # except: print('no photo')
            # Parse the message text with spacy
            doc = nlp(message.text)
            # Loop through the entities in the doc
            for ent in doc.ents:
                #print("ent.text, ent.label_")
                #print(ent.text, ent.label_)
                # Check if the entity is a date, an address or a category (such as PERSON, ORG, etc.)
                if ent.label_ in ["DATE"]: #, "GPE", "LOC", "PERSON", "ORG"]:
                    # Print the entity text and label
                    print(ent.text, ent.label_)
                    
# Run the async function using asyncio.run()
asyncio.run(get_messages(),debug=True)