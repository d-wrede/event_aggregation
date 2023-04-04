import json
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import dict_to_timestamp, check_timestamps
import dateparser.search
import dateparser
from dateparser_data.settings import default_parsers

json_filename = '/Users/danielwrede/Documents/read_event_messages/telegram_messages.json'


def main():

    # read messages from json file
    with open(json_filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    default_parsers.reverse()

    settings = {
        # Order to prioritize when parsing ambiguous dates (Day, Month, Year)
        'DATE_ORDER': 'DMY',
        'PREFER_DATES_FROM': 'future',  # Prefer dates from the future
        'STRICT_PARSING': False,  # Allow for approximate parsing
        'NORMALIZE': True,  # Normalize whitespace and remove extra spaces within the date string
        'RETURN_AS_TIMEZONE_AWARE': False,  # Return timezone-aware datetime objects
        'DEFAULT_LANGUAGES': ["de"],
        # 'LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD': 0.5,
        'PARSERS': default_parsers
        # 'RETURN_TIME_AS_PERIOD': True,
        # 'PREFER_DAY_OF_MONTH': 'first',
    }

    
    count_dates = 0
    count_messages = 0
    count_parses = 0

    # timestamp extraction
    for message in messages[:]:
        
        # and 'approved' not in message['timestamps']['comment']:
        if 'message' in message and message['message'] != '':
            count_messages += 1
            date_message = dateparser.parse(message['date'])
            #print("message: ", message['message'])

            # extract timestamps as list of dicts with date and time
            time_matches = extract_timestamp(
                message['message'])

            # convert dicts to timestamps:
            # {year: YYYY, month: MM, day: DD, hour: HH, minute: MM} -> datetime
            for datedict in time_matches:
                datedict['timestamp'] = dict_to_timestamp(datedict.copy())

            # parse with dateparser
            settings['RELATIVE_BASE'] = date_message
            parsedstamps = dateparser.search.search_dates(
                message['message'], languages=['de'], settings=settings)

            # Set seconds to 0 for each parsed datetime object
            parsedstamps_no_seconds = []
            if parsedstamps:
                # step through parsedstamps and remove seconds
                for date_str, date_obj in parsedstamps:
                    date_obj = date_obj.replace(second=0)
                    parsedstamps_no_seconds.append((date_str, date_obj))
                parsedstamps = parsedstamps_no_seconds

                #check if parsedstamps are in timestamps
                # timestamps_refactored = [stampdict['timestamp']
                #                          for stampdict in times_refactored]
                timestamps = [
                    (stampdict['matching_substring'], stampdict['timestamp']) for stampdict in time_matches]

                check_timestamps(timestamps, parsedstamps)


            # add timestamps to message dict
            message.setdefault('timestamps', {})
            message['timestamps'] = timestamps
            message.setdefault('parsedstamps', {})
            message['parsedstamps'] = parsedstamps

            if time_matches is not None:
                count_dates += 1
            if parsedstamps is not None:
                count_parses += 1

    # sort messages by timestamp
    messages.sort(key=lambda x: str(
        x.get('timestamps', float('inf'))))

    print(
        f'found {count_dates} dates and {count_parses} parses in {len(messages)} messages')
    # save messages with timestamps
    # with open('new_message_list.json', 'w', encoding='utf-8') as f:
    #     json.dump(messages, f, indent=4, ensure_ascii=False)

    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                if message['timestamps']:
                    timestamps_sorted = sorted(message['timestamps'], key=lambda x: str(
                        x if x is not None else float('inf')))
                    f.write(str(timestamps_sorted) + '\n')
                    print('timestamps: ', timestamps_sorted)
                if message['parsedstamps']:
                    parsedstamps_sorted = sorted(message['parsedstamps'], key=lambda x: x[1])
                    f.write(str(parsedstamps_sorted) + '\n\n')
                    print('parsedstamps: ', parsedstamps_sorted)





if __name__ == "__main__":
    main()
