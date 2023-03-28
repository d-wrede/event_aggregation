import json
from src.extract_timestamp import extract_timestamp, filter_string
import dateparser.search
import dateparser
from dateparser_data.settings import default_parsers

json_filename = '/Users/danielwrede/Documents/read_event_messages/telegram_messages.json' #'telegram_messages.json'


def main():
    # Opening messages file
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
    
    # blacklist for parsedstamps
    blacklist = ['mit', 'in die', 'in die', 'und die', 'so', 'die', '40',
                 '2-3', '1,5', 'jahren', 'und mit', 'mit einem', '18', '14€/12€', '10€/8€', '15€ / 20€', '0163', '60€', '6 mit', '12-22']
    # list of useful tempex: In drei Stunden, Am Sa, diese woche, Fr 18 Uhr, 
    count_dates = 0
    count_messages = 0
    count_parses = 0
    # timestamp extraction
    for message in messages[:20]:
        if 'message' in message and message['message'] != '':# and 'approved' not in message['timestamps']['comment']:
            count_messages += 1
            date_message = dateparser.parse(message['date'])
            #print(date_message)
            
            timestamps = extract_timestamp(message['message'])

            # if timestamps is None:
                # try to parse with dateparser
            settings['RELATIVE_BASE'] = date_message
            parsedstamps = dateparser.search.search_dates(message['message'], languages=[
                                        'de'], settings=settings)
            
            """
            timestamps:  (datetime.datetime(2023, 4, 28, 1, 5), datetime.datetime(2023, 5, 1, 15, 15), '') 
            parsedstamps:  [('28.04.-01.05', datetime.datetime(2023, 4, 28, 1, 5)), ('mit', datetime.datetime(2023, 3, 29, 0, 0)), ('in die', datetime.datetime(2023, 3, 28, 0, 0)), ('Jetzt', datetime.datetime(2023, 3, 23, 11, 39, 21)), ('Am Samstag', datetime.datetime(2023, 3, 25, 0, 0)), ('um 15.15 Uhr am', datetime.datetime(2023, 3, 23, 15, 15)), ('Die', datetime.datetime(2023, 3, 28, 0, 0)), ('in die', datetime.datetime(2023, 3, 28, 0, 0)), ('und die', datetime.datetime(2023, 3, 28, 0, 0)), ('mit', datetime.datetime(2023, 3, 29, 0, 0)), ('mit', datetime.datetime(2023, 3, 29, 0, 0)), ('so', datetime.datetime(2023, 3, 26, 0, 0))]"""

            # check if parsedstamps are in timestamps
            if timestamps is not None and parsedstamps is not None:
                switch_message = False
                for stamp in parsedstamps:
                    tup_ts = tuple(dt.strftime('%Y-%m-%d')
                                   for dt in timestamps[:-1])
                    tup_time = tuple(dt.strftime('%H:%M')
                                     for dt in timestamps[:-1])
                    if stamp[1].strftime('%Y-%m-%d') not in tup_ts and stamp[0].lower() not in blacklist:
                        print("parsedstamp: ", stamp)
                        switch_message = True
                        #print("-"*50)
                        #print(message['message'], '\n')
                    if stamp[1].strftime('%H:%M') not in tup_time and stamp[0].lower() not in blacklist:
                        print("parsedstamp: ", stamp)
                        switch_message = True
                if switch_message:
                    print("timestamps: ", timestamps)
                    print("")
                    switch_message = False



            
            #print("timestamps: ", timestamps, '\n')
            #print("parsedstamps: ", parsedstamps, '\n')
            # add timestamps to message dict
            message.setdefault('timestamps', {})
            message['timestamps'] = timestamps
            message.setdefault('parsedstamps', {})
            message['parsedstamps'] = parsedstamps
            
            


            if timestamps is not None:
                count_dates += 1
            if parsedstamps is not None:
                count_parses += 1

        # if count_messages == 5:
        #     break
    
    # sort messages by timestamp
    messages.sort(key=lambda x: str(
        x.get('timestamps', float('inf'))))

    print(f'found {count_dates} dates and {count_parses} parses in {len(messages)} messages')
    # save messages with timestamps
    # with open(json_filename, 'w', encoding='utf-8') as f:
    #     json.dump(messages, f, indent=4, ensure_ascii=False)
    exit()
    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                #print(message['timestamps'])
                f.write(str(list(message['timestamps']).sort(key=lambda x: str(
                    x if x is not None else float('inf')))) + '\n')
                #print(message['parsedstamps'])
                f.write(str(message['parsedstamps'].sort(
                    key=lambda x: x[1])) + '\n\n')
                
                # write comment if available
                # if message['timestamps']['comment'] is not None:
                #     f.write(str(message['timestamps']['comment']) + '\n')
                #f.write('-'*50 + '\n')
                #f.write(filter_string(message['message']) + '\n\n')

    # further terms to be extracted:
    # - sender/author spaCy?
    # - place
    # - category
    # - URL


if __name__ == "__main__":
    main()
