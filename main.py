import json
from src.extract_timestamp import extract_timestamp


def main():
    # Opening messages file
    with open('telegram_messages.json', 'r', encoding='utf-8') as f:
        messages = json.load(f)

    for message in messages:
        stamps = extract_timestamp(message['message'])
        print(stamps)


if __name__ == "__main__":
    main()
