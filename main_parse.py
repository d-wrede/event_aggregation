import json
from src.extract_timestamp import extract_timestamp, filter_string
import dateparser.search

json_filename = 'telegram_messages.json'


def main():
    # Opening messages file
    with open(json_filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    # Custom settings for dateparser
    settings = {
        # Order to prioritize when parsing ambiguous dates (Day, Month, Year)
        'DATE_ORDER': 'DMY',
        'PREFER_DATES_FROM': 'future',  # Prefer dates from the future
        'STRICT_PARSING': False,  # Allow for approximate parsing
        'NORMALIZE': True,  # Normalize whitespace and remove extra spaces within the date string
        'RETURN_AS_TIMEZONE_AWARE': False,  # Return timezone-aware datetime objects
    }

    count_dates = 0
    count_messages = 0
    # timestamp extraction
    for message in messages[:]:
        if 'message' in message and message['message'] != '':# and 'approved' not in message['timestamps']['comment']:
            count_messages += 1
            timestamps = dateparser.search.search_dates(message['message'], languages=['de'], settings=settings)
            
            #extract_timestamp(message['message'])
            print(message['message'])
            print("timestamps: ", timestamps, '\n')
            # add timestamps to message dict
            message.setdefault('timestamps', {})
            message['timestamps'] = timestamps
            
            if timestamps is not None:
                count_dates += 1

        # if count_messages == 5:
        #     break
    
    # sort messages by timestamp
    messages.sort(key=lambda x: str(
        x.get('timestamps', float('inf'))))

    print(f'found {count_dates} dates in {len(messages)} messages')
    # save messages with timestamps
    # with open(json_filename, 'w', encoding='utf-8') as f:
    #     json.dump(messages, f, indent=4, ensure_ascii=False)

    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                f.write(str(message['timestamps']) + '\n')
                
                # write comment if available
                # if message['timestamps']['comment'] is not None:
                #     f.write(str(message['timestamps']['comment']) + '\n')
                f.write('-'*50 + '\n')
                f.write(filter_string(message['message']) + '\n\n')

    # further terms to be extracted:
    # - sender/author spaCy?
    # - place
    # - category
    # - URL


if __name__ == "__main__":
    main()
