import json
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import (
    dateparser_vs_ownparser,
    interpret_dates,
    get_min_date,
)
from src.extract_place import extract_place


json_filename = (
    "/Users/danielwrede/Documents/read_event_messages/telegram_messages.json"
)


def main():
    # read messages from json file
    with open(json_filename, "r", encoding="utf-8") as f:
        messages = json.load(f)

    count_dates = 0
    count_messages = 0

    # timestamp extraction
    for message in messages[:]:
        # and 'approved' not in message['timestamps']['comment']:
        if "message" not in message or message["message"] == "":
            continue

        count_messages += 1

        # extract timestamps as list of dicts with date and time using own parser
        time_matches = extract_timestamp(message["message"])

        # interpret dates by connecting date and time
        interpreted_dates = interpret_dates(time_matches)

        # add timestamps to message dict
        message.setdefault("timestamps", [])
        message["timestamps"] = interpreted_dates

        # compare dateparser with own parser results
        # dateparser_vs_ownparser(message, time_matches)

        # further terms to be extracted:
        # - sender/author
        # - place
        place, topic, misc = extract_place(message["message"])
        print("place: ", place)
        print("topic: ", topic)
        print("misc: ", misc)
        # - category

        if time_matches is not None:
            count_dates += 1

        # print message to file, if it contains a timestamp
        if time_matches:
            with open("file.txt", "a", encoding="utf-8") as f:
                for match in time_matches:
                    timestamp = ""
                    if match["date1"]:
                        timestamp += match["date1"].strftime("%Y-%m-%d")
                    if match["clock1"]:
                        timestamp += " " + match["clock1"].strftime("%H:%M")
                    if match["date2"]:
                        timestamp += " - " + match["date2"].strftime("%Y-%m-%d")
                    if match["clock2"]:
                        timestamp += " " + match["clock2"].strftime("%H:%M")

                    f.write(
                        f'{match["matching_substring"]}: {timestamp}, priority: {match["priority"]}, pattern_type: {match["pattern_type"]}\n'
                    )
                f.write(str(filter_string(message["message"])) + "\n\n")

    print(f"found {count_dates} dates in {len(messages)} messages")

    # sort messages by timestamp
    messages.sort(key=get_min_date)

    # save messages with timestamps
    with open("new_message_list.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
