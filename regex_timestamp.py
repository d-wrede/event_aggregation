import json
import re
import datetime

def filter_string(string):
    string = re.sub(
        r'[^\da-zA-Z\säöüÄÖÜß/,.-]|\.(?=(\s|$))', '', string)
    string = string.replace('--', '-')
    string = re.sub(r'[\t\xa0]+', ' ', string)
    string = re.sub(r'[\n]+[\s]+[\n]+', '\n', string)
    # TODO: consider to allow stops (,) as well
    return string

# Opening messages file
with open('telegram_messages.json', 'r', encoding='utf-8') as f:
    messages = json.load(f)

# regex date patterns
date_patterns = [
    # regex structured with groups
    r"\b(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]*(Jan|Feb|März|Maerz|April|Ma[iy]|Jun|Jul|Aug|Sep|Okt|Nov|Dez)[a-z]*[-.\s/]*(20)?(2[2-5])?\b",
    r"\b(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+0?([1-9]|1[0-2])[-.\s/]+(20)?(2[2-5])?\b"
]
ddate_patterns = [
    r"\b(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+(Jan|Feb|März|Maerz|April|Ma[iy]|Jun|iul|Aug|Sep|Okt|Nov|Dez)[a-z]*[-.\s/]*(20)?(2[2-5])?\b",
    r"\b(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+(0?[1-9]|[1-2][0-9]|3[01])[-.\s/]+0?([1-9]|1[0-2])[-.\s/]+(20)?(2[2-5])?\b"
]
clock_pattern = r"(?<!\.)\b([0-1]?[0-9]|2[0-3])[:.]([0-5][0-9])\b(?!\.)"

month_dict = {
    'Jan': 1,  'Feb': 2,  'März': 3, 'Maerz': 3, 'April': 4,
    'Mai': 5,  'Jun': 6,  'Jul': 7,  'Aug': 8,   'Sep': 9,
    'Okt': 10, 'Nov': 11, 'Dez': 12
}

# timestamp extraction
for message in messages[:]:
    if 'message' not in message:
        continue
    dateset = set()
    clockset = set()
    print("message: ", repr(filter_string(message['message'][0:100])))

    # initializing variable
    start_date = None
    end_date = None

    # retrieve dates from message
    for i, date_pattern in enumerate(date_patterns):
        dates = re.findall(
            date_pattern, message['message'], flags=re.IGNORECASE)
        ddates = re.findall(
            ddate_patterns[i], message['message'], flags=re.IGNORECASE)

        for date in dates:
            day = date[0]
            month = date[1]
            year = date[3]
            # if year is empty, set it to current year
            if year == '':
                now = datetime.datetime.now()
                year = now.year % 100

            if month.isalpha():
                month = month_dict[month]
            
            dateset.add((day, month, year))

        # dates like DD. - DD. MM. YYYY
        for ddate in ddates:
            day1 = ddate[0]
            day2 = ddate[1]
            month = ddate[2]
            year = ddate[4]
            # if year is empty, set it to current year
            if year == '':
                now = datetime.datetime.now()
                year = now.year % 100

            if month.isalpha():
                month = month_dict[month]
            
            start_date = datetime.date(2000 + int(year), int(month), int(day1))
            end_date = datetime.date(2000 + int(year), int(month), int(day2))

    # retrieve clocks from message
    clocks = re.findall(clock_pattern, message['message'], flags=re.IGNORECASE)
    clockset.update(clocks)
    for clock in clocks:
        hour = clock[0]
        minute = clock[1]

    # join results
    if dateset:
        if start_date is None or end_date is None:
            # Convert date strings to datetime.date objects)
            dates = [datetime.date(2000 + int(year), int(month), int(day))
                     for year, month, day in dateset]

            # Find min and max dates
            start_date = min(dates)
            end_date = max(dates)

        try:
            # Convert time strings to datetime.time objects
            times = [datetime.time(hour=int(hour), minute=int(minute))
                    for hour, minute in clockset]
            # Find min and max times
            start_time = min(times)
            end_time = max(times)
        except ValueError:
            if (end_date - start_date) > datetime.timedelta(days=1):
                print("setting start_time and end_time:")
                start_time = datetime.time(12, 00)
                end_time = datetime.time(12, 00)        

        print(
            f"Start: on the {start_date} at {start_time} o'clock\nEnd: on the {end_date} at {end_time} o'clock.")
        startstamp = datetime.datetime.combine(start_date, start_time)
        endstamp = datetime.datetime.combine(end_date, end_time)
    else:
        print("No dates or clocks found.")
    print(' ')
