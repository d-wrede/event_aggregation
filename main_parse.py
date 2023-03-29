import json
from src.extract_timestamp import extract_timestamp, filter_string
import dateparser.search
import dateparser
from dateparser_data.settings import default_parsers
import re

# 'telegram_messages.json'
json_filename = '/Users/danielwrede/Documents/read_event_messages/telegram_messages.json'


def main():

    def match_patterns(blackregexlist, string):
        for blackregex in blackregexlist:
            if re.search(blackregex, string, flags=re.IGNORECASE):
                return True
        return False

    # blacklist for parsedstamps
    blacklist = ['mit', 'in die', 'in die', 'und die', 'so', 'die', '40',
                 '2-3', '1,5', 'jahren', 'und mit', 'mit einem', '18', '14€/12€', '10€/8€', '15€ / 20€', '0163', '60€', '6 mit', '12-22']
    blackregexlist = [r'^\d{2}$', r'^\d{1,2}-\d{1,2}$',
                      r'^\d{1,2}€[/-]\d{2}€', r'^\d{1-3}$', r'(€|Euro)']

    def check_timestamps(timestamps, parsedstamps, blacklist=blacklist, blackregexlist=blackregexlist):
        """Check if parsedstamps are in timestamps, on the blacklist or blackregexlist. If not print them."""
        if timestamps is not None and parsedstamps is not None:
            switch_message = False
            for stamp in parsedstamps:
                # generate date and time tuples for comparison
                tups_ts = tuple(dt.strftime(
                    '%Y-%m-%d') for dt in timestamps[:-1] if dt is not None)  # if dt is not None else None
                tups_time = tuple(dt.strftime(
                    '%H:%M') for dt in timestamps[:-1] if dt is not None)
                # tups_fulltimestamp = tuple(
                #     stamp for stamp in timestamps[:-1] if stamp is not None)

                if match_patterns(blackregexlist, stamp[0]):
                    continue
                if stamp[1].strftime('%Y-%m-%d') in tups_ts:
                    continue
                if stamp[1].year > 2025:
                    continue
                if timestamps[0] is not None:
                    if (min(timestamps[:-1]) < stamp[1] < max(timestamps[:-1])):
                        continue
                if len(tups_ts) > 0:
                    if min(tups_ts) < stamp[1].strftime('%Y-%m-%d') < max(tups_ts):
                        continue
                if len(tups_time) > 0:
                    if min(tups_time) < stamp[1].strftime('%H:%M') < max(tups_time):
                        continue
                if stamp[0].lower() in blacklist:
                    continue
                if stamp[1].strftime('%H:%M') in tups_time:
                    continue
                if stamp[0].lower() in blacklist:
                    continue

                # if stamp[1].strftime('%Y-%m-%d') == '2023-03-31' or stamp[0] == '31.3.23':
                #     print("tup_ts: ", tup_ts)
                #     print("stamp: ", stamp)
                print("parsedstamp: ", stamp)
                switch_message = True

            if switch_message:
                print("timestamps: ", timestamps)
                print("")
                switch_message = False

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

    # list of useful tempex: In drei Stunden, Am Sa, diese woche, Fr 18 Uhr,
    count_dates = 0
    count_messages = 0
    count_parses = 0
    # timestamp extraction
    for message in messages[:]:
        # and 'approved' not in message['timestamps']['comment']:
        if 'message' in message and message['message'] != '':
            count_messages += 1
            date_message = dateparser.parse(message['date'])
            # print(date_message)

            timestamps = extract_timestamp(message['message'])

            # if timestamps is None:
            # try to parse with dateparser
            settings['RELATIVE_BASE'] = date_message
            parsedstamps = dateparser.search.search_dates(
                message['message'], languages=['de'], settings=settings)

            # Set seconds to 0 for each parsed datetime object
            parsedstamps_no_seconds = []
            if parsedstamps:
                for date_str, date_obj in parsedstamps:
                    date_obj = date_obj.replace(second=0)
                    parsedstamps_no_seconds.append((date_str, date_obj))
                parsedstamps = parsedstamps_no_seconds

                # check if parsedstamps are in timestamps
                check_timestamps(timestamps, parsedstamps,
                                blacklist, blackregexlist)

            # print("timestamps: ", timestamps, '\n')
            # print("parsedstamps: ", parsedstamps, '\n')
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

    print(
        f'found {count_dates} dates and {count_parses} parses in {len(messages)} messages')
    # save messages with timestamps
    # with open(json_filename, 'w', encoding='utf-8') as f:
    #     json.dump(messages, f, indent=4, ensure_ascii=False)
    exit()
    with open('file.txt', 'w', encoding='utf-8') as f:
        for message in messages:
            if 'message' in message and message['message'] != '':
                # print(message['timestamps'])
                f.write(str(list(message['timestamps']).sort(key=lambda x: str(
                    x if x is not None else float('inf')))) + '\n')
                # print(message['parsedstamps'])
                f.write(str(message['parsedstamps'].sort(
                    key=lambda x: x[1])) + '\n\n')

                # write comment if available
                # if message['timestamps']['comment'] is not None:
                #     f.write(str(message['timestamps']['comment']) + '\n')
                # f.write('-'*50 + '\n')
                # f.write(filter_string(message['message']) + '\n\n')

    # further terms to be extracted:
    # - sender/author spaCy?
    # - place
    # - category
    # - URL


if __name__ == "__main__":
    main()
