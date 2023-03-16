# Import libraries
import asyncio, configparser
from telethon import TelegramClient
import spacy

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

chat_id = int(config['Channels']['Ecstatic_Dance'])

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

    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if 'dance'.lower() in dialog.name.lower():
            print(f"{dialog.name}, {dialog.id}") 
        # if dialog.entity.phone:
        #     print("\tphone: ", dialog.entity.phone)
    
    
    group_entity = await client.get_input_entity(chat_id)
    print("channel: ", group_entity.stringify())
    connection_status = client.is_connected()
    print("connection_status: ", connection_status)

    messages = await client.get_messages(group_entity, limit = 10)
    
    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if dialog.id == chat_id:
            print(f"{dialog.name}, {dialog.id}")

    # Loop through the messages
    for counter,message in enumerate(messages):
        # Check if the message is text
        print("\nmessage number: ", counter)

        if message.text:
            # Print the message text
            print(message.text)

            # Parse the message text with spacy
            doc = nlp(message.text)
            # Loop through the entities in the doc
            for ent in doc.ents:
                # Check if the entity is a date
                if ent.label_ in ["DATE"]:
                    # Print the entity text and label
                    print(ent.text, ent.label_)
                    
# Run the async function using asyncio.run()
asyncio.run(get_messages(),debug=True)