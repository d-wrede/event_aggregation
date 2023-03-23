# Import libraries
import asyncio, configparser
from telethon import TelegramClient
import spacy
from datetime import datetime
import json

# channel search term
search_term = 'dance'
json_filename = 'telegram_messages.json'

# Reading Configs
config = configparser.ConfigParser()
config.read("config.ini")

# Setting configuration values
api_id = config['Telegram']['api_id']
api_hash = str(config['Telegram']['api_hash'])
phone = config['Telegram']['phone']
username = config['Telegram']['username']
channel = 'Ecstatic_Dance'
chat_id = int(config['Channels'][channel])
channel = 'Ecstatic_Dance'
chat_id = int(config['Channels'][channel])

# Create a client object with your credentials
client = TelegramClient(username, api_id, api_hash)

# Define an async function to get messages from channel
async def get_messages():
    # Start the client
    await client.start()
    me = await client.get_me()
    print(f"{me.first_name} {me.last_name}")
    print("id ", me.id)
    print("phone +", me.phone)

    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if search_term.lower() in dialog.name.lower():
            print(f"{dialog.name}, {dialog.id}") 
            # print phone number, if available    
            if hasattr(dialog.entity, 'phone'):
                print("\tphone: +", dialog.entity.phone)
            # print phone number, if available    
            if hasattr(dialog.entity, 'phone'):
                print("\tphone: +", dialog.entity.phone)
    
    group_entity = await client.get_input_entity(chat_id)
    print("channel: ", group_entity.stringify())
    connection_status = client.is_connected()
    print("connection_status: ", connection_status)

    messages = await client.get_messages(group_entity, limit = 50)
    authors = [None]*len(messages)

    # get sender names
    for i, message in enumerate(messages):
        # get user_id from message
        user_id = message.from_id.user_id
        # Get the User entity associated with the user ID
        user_entity = await client.get_entity(user_id)

        # Get the first name and last name of the user
        first_name = user_entity.first_name
        last_name = user_entity.last_name
        authors[i] = {'first_name': first_name, 'last_name': last_name}

    # Get the Channel entity associated with the chat-/channel ID
    channel_entity = await client.get_entity(chat_id)
    # Get the name of the channel
    channel_name = channel_entity.title    

    # print contact details
    dialogs = await client.get_dialogs()
    for dialog in dialogs:
        if dialog.id == chat_id:
            print(f"{dialog.name}, {dialog.id}")


    messages_list = [message.to_dict() for message in messages]
    # add authors and channel name to messages
    for message, author in zip(messages_list, authors):
        message['from_id'].update(author)
        message['channel_name'] = channel_name
    # save messages to json file
    with open('telegram_messages.json', 'w', encoding='utf-8') as f:
        json.dump(messages_list, f, indent=4, sort_keys=True, default=str)

                    
# Run the async function using asyncio.run()
asyncio.run(get_messages(),debug=True)