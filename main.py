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
    rake_keywords,
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
    # with for loop each message with timestamp for spacy NER and RAKE
    filtered_messages = [message for message in messages[:number_of_messages]
        if "message" in message and message["message"] != "" and 'timestamps' in message and len(message['timestamps']) and isinstance(message['timestamps'], list) and message['timestamps'][0]['date1'] is not None]
    for message in filtered_messages:
        cleaned_message = filter_string(message["message"][:first_letters])
        place, topic, misc = spacy_ner(cleaned_message)
        message.setdefault("topic_suggestions", {})
        message["topic_suggestions"]["spacy_NER"] = filter_keywords(
            place + topic + misc
        )
        keywords_rake = filter_keywords(rake_keywords(cleaned_message))
        message["topic_suggestions"]["rake_keywords"] = keywords_rake

    # all messages in one batch for tf-IDF, LDA and NMF topic modeling
    # Clean and preprocess the texts
    cleaned_texts = [
        filter_string(message["message"][:first_letters])
        for message in filtered_messages
    ]
    # extract_topic(cleaned_texts)
    tf_IDF_keywords = tf_IDF(cleaned_texts)
    LDA_keywords = LDA_topic_modeling(cleaned_texts)
    LDA_keywords = sort_keywords_by_input_order(LDA_keywords, cleaned_texts)

    NMF_keywords = NMF_topic_modeling(cleaned_texts)
    NMF_keywords = sort_keywords_by_input_order(NMF_keywords, cleaned_texts)
    for i, message in enumerate(filtered_messages):
        if "message" not in message or message["message"] == "":
            continue
        message["topic_suggestions"]["tf_IDF"] = filter_keywords(tf_IDF_keywords[i])
        message["topic_suggestions"]["LDA"] = filter_keywords(LDA_keywords[i])
        message["topic_suggestions"]["NMF"] = filter_keywords(NMF_keywords[i])

    # sort keywords to (most probable) common topics
    for message in filtered_messages:
        common_topics = find_common_topics(message["topic_suggestions"], filter_string(message["message"][:first_letters]))
        common_topics = filter_keywords(common_topics)
        print("message: ", filter_string(message["message"][:first_letters]))

        # print timestamps
        timestamps = message["timestamps"]
        for stamp in timestamps:
            if stamp["date1"]: print("date1: ", stamp["date1"])
            if stamp["clock1"]: print("clock1: ", stamp["clock1"])
            if stamp["date2"]: print("date2: ", stamp["date2"])
            if stamp["clock2"]: print("clock2: ", stamp["clock2"])
            print("---")

        if 'Thema:' in message["message"]:
            index_position = message["message"].find('Thema:') + len('Thema:')
            newline_position = message["message"][index_position:].find('\n') + index_position
            if newline_position == -1:
                newline_position = len(message["message"])
            topic = message["message"][index_position:newline_position].strip()
            print('the topic is: ', topic)
            message["topic_suggestions"]["common_topics"] = topic
        else:
            message["topic_suggestions"]["common_topics"] = common_topics

            print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])
            print("rake_keywords: ", message["topic_suggestions"]["rake_keywords"])
            print("## later added ##")
            print("tf_IDF: ", message["topic_suggestions"]["tf_IDF"])
            print("LDA: ", message["topic_suggestions"]["LDA"])
            print("NMF: ", message["topic_suggestions"]["NMF"])
        print("common topics: ", common_topics)
        print("")


    # sort messages by timestamp
    filtered_messages.sort(key=get_min_date)

    # topic extraction algorithm evaluation
    evaluate_topic_extraction(filtered_messages)

    print("word frequency")
    
    # word_frequencies = []
    # for message in filtered_messages:
    #     word_frequencies.extend(word_frequency(message["topic_suggestions"]["common_topics"]))
    
    # Sort the combined list by frequency (descending order)
    # sorted_list = sorted(word_frequencies, key=lambda x: x[1], reverse=True)

    # Print the sorted list
    # for word, freq in sorted_list:
    #     print(f"{word}: {freq}")

    
    
    # topiclist = []
    # for message in filtered_messages:
    #     topicdict = {
    #         "message": message["message"][:200],
    #         'id': message["id"],
    #         'common_topics': message["topic_suggestions"]["common_topics"],
    #     }
    #     topiclist.append(topicdict)
    # with open("topics.json", "r", encoding="utf-8") as f:
    #     json.dump(topiclist, f, indent=4, ensure_ascii=False)


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
