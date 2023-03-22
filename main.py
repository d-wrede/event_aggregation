import json
from src.extract_timestamp import extract_timestamp


def main():
    # Opening messages file
    with open('telegram_messages.json', 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    for message in messages:
        if 'message' in message:
            timestamps = extract_timestamp(message['message'])
            print(message['message'][0:70])
            print(timestamps, '\n')
    
    # further terms to be extracted:
    # - sender/author
    # - place
    # - category



if __name__ == "__main__":
    main()
