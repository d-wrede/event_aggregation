import json
import pandas as pd
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import (
    dateparser_vs_ownparser,
    interpret_dates,
    get_min_date,
)
from src.extract_topic import (
    spacy_ner,
    rake,
    cluster_messages,
    tf_IDF,
    LDA_topic_modeling,
    sort_keywords_by_input_order,
    NMF_topic_modeling,
    find_common_topics,
    remove_stopwords,
    filter_keywords,
    evaluate_topic_extraction,
    word_frequency,
    extract_keywords, store_keywords_in_messages, extract_common_topics,
    check_thema,
    get_thema_topic,
    set_thema_in_messages
)



json_filename = (
    "/Users/danielwrede/Documents/read_event_messages/telegram_messages.json"
)

number_of_messages = 600
first_letters = 500

def main():
    # read messages from json file
    with open(json_filename, "r", encoding="utf-8") as f:
        messages = json.load(f)

    count_dates = 0
    count_messages = 0

    # timestamp extraction
    for message in messages[:number_of_messages]:
        # and 'approved' not in message['timestamps']['comment']:
        if "message" not in message or message["message"] == "":
            continue

        count_messages += 1

        # extract timestamps as list of dicts with date and time using own parser
        time_matches = extract_timestamp(message["message"])

        # interpret dates by connecting date and time
        interpreted_dates = interpret_dates(time_matches)

        message.setdefault("timestamps", [])
        if len(time_matches):
            # add timestamps to message dict
            message["timestamps"] = interpreted_dates
            count_dates += 1
            pass

        # compare dateparser with own parser results
        # dateparser_vs_ownparser(message, time_matches)

        # print message to file, if it contains a timestamp
        # if len(time_matches):
        #     with open("file.txt", "a", encoding="utf-8") as f:
        #         for match in time_matches:
        #             timestamp = ""
        #             if match["date1"]:
        #                 timestamp += match["date1"].strftime("%Y-%m-%d")
        #             if match["clock1"]:
        #                 timestamp += " " + match["clock1"].strftime("%H:%M")
        #             if match["date2"]:
        #                 timestamp += " - " + match["date2"].strftime("%Y-%m-%d")
        #             if match["clock2"]:
        #                 timestamp += " " + match["clock2"].strftime("%H:%M")

        #             f.write(
        #                 f'{match["matching_substring"]}: {timestamp}, priority: {match["priority"]}, pattern_type: {match["pattern_type"]}\n'
        #             )
        #         f.write(str(filter_string(message["message"])) + "\n\n")

    print(f"found {count_dates} dates in {len(messages)} messages")


    ### extract topic ###
    # list messages with timestamps and message text
    filtered_messages = [message for message in messages[:number_of_messages]
        if "message" in message and message["message"] != "" and 'timestamps' in message and len(message['timestamps']) and isinstance(message['timestamps'], list) and message['timestamps'][0]['date1'] is not None]

    # add topic_suggestions key to each message
    for message in filtered_messages:
        message.setdefault("topic_suggestions", {})

    set_thema_in_messages(filtered_messages)

    # Clean and preprocess the texts
    cleaned_texts_with_indices = [
    (idx, filter_string(message["message"][:first_letters]))
    for idx, message in enumerate(filtered_messages)
    if "common_topics" not in message["topic_suggestions"]
    ]

    # get keywords for each message
    spacy_keywords, rake_keywords, tf_IDF_keywords, LDA_keywords, NMF_keywords = extract_keywords([text for _, text in cleaned_texts_with_indices])
    store_keywords_in_messages(filtered_messages, spacy_keywords, rake_keywords, tf_IDF_keywords, LDA_keywords, NMF_keywords, cleaned_texts_with_indices)
    extract_common_topics(filtered_messages, first_letters)

    # sort messages by timestamp
    filtered_messages.sort(key=get_min_date)

    # topic extraction algorithm evaluation
    evaluate_topic_extraction(filtered_messages)
    
    # safe in readable format
    with open("file.txt", "w", encoding="utf-8") as f:
        for message in filtered_messages:
            f.write(f'### message ###\n{filter_string(message["message"])[:500]} \n---\n')
            timestamps = message['timestamps']
            for stamp in timestamps:
                timestamp = ""
                if stamp["date1"]:
                    timestamp += stamp["date1"]
                if stamp["clock1"]:
                    timestamp += " " + stamp["clock1"]
                if stamp["date2"]:
                    timestamp += " - " + stamp["date2"]
                if stamp["clock2"]:
                    timestamp += " " + stamp["clock2"]

                f.write(f'timestamp: {timestamp} \n')
            # topics
            f.write(f'topics: {message["topic_suggestions"]["common_topics"][:6]}\n\n')

    # save messages with timestamps
    with open("new_message_list.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
