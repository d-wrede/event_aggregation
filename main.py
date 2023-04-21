import json
from src.extract_timestamp import extract_timestamp, filter_string
from src.organize_timestamps import (
    dateparser_vs_ownparser,
    interpret_dates,
    get_min_date,
)
from src.extract_place import spacy_ner, rake_keywords
from src.extract_topic import (
    cluster_messages,
    tf_IDF,
    LDA_topic_modeling,
    sort_keywords_by_input_order2,
    NMF_topic_modeling,
    find_common_topics,
    remove_stopwords,
    filter_keywords,
)


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
    for message in messages[:6]:
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

        cleaned_message = remove_stopwords(filter_string(message["message"]))
        place, topic, misc = spacy_ner(cleaned_message)
        # print("message: ", filter_string(message["message"]))
        # print("place: ", place)
        # print("topic: ", topic)
        # print("misc: ", misc)
        message.setdefault("topic_suggestions", {})
        message["topic_suggestions"]["spacy_NER"] = filter_keywords(place + topic + misc)
        # - category
        keywords_rake = filter_keywords(rake_keywords(cleaned_message))
        message["topic_suggestions"]["rake_keywords"] = keywords_rake
        # LDA_keywords =  LDA_topic_modeling(cleaned_message)
        # message["topic_suggestions"]["LDA_keywords"] = LDA_keywords
        # LDA_topic_modeling(filter_string(message["message"]))

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

    # extract topic
    # Clean and preprocess the texts
    cleaned_texts = [
        remove_stopwords(filter_string(message["message"]))
        for message in messages[:6]
        if "message" in message and message["message"] != ""
    ]
    # extract_topic(cleaned_texts)
    tf_IDF_keywords = tf_IDF(cleaned_texts)
    LDA_keywords =  LDA_topic_modeling(cleaned_texts)
    LDA_keywords = sort_keywords_by_input_order2(LDA_keywords, cleaned_texts)
    # for cleaned_text, keywords in zip(cleaned_texts, LDA_keywords):
    #     print("cleaned_text: ", cleaned_text)
    #     print("keywords: ", keywords)
    #     print("")
    
    NMF_keywords = NMF_topic_modeling(cleaned_texts)
    NMF_keywords = sort_keywords_by_input_order2(NMF_keywords, cleaned_texts)
    for i, message in enumerate(messages[:5]):
        if "message" in message and message["message"] != "":
            message["topic_suggestions"]["tf_IDF"] = filter_keywords(tf_IDF_keywords[i])
            message["topic_suggestions"]["LDA"] = filter_keywords(LDA_keywords[i])
            message["topic_suggestions"]["NMF"] = filter_keywords(NMF_keywords[i])

    for message in messages[:5]:
        common_topics = find_common_topics(message['topic_suggestions'])
        common_topics = filter_keywords(common_topics)
        message['topic_suggestions']['common_topics'] = common_topics
        print("message: ", filter_string(message['message']))
        print("spacy_NER: ", message['topic_suggestions']['spacy_NER'])
        print("rake_keywords: ", message['topic_suggestions']['rake_keywords'])
        print("## later added ##")

        print("tf_IDF: ", message['topic_suggestions']['tf_IDF'])
        print("LDA: ", message['topic_suggestions']['LDA'])
        print("NMF: ", message['topic_suggestions']['NMF'])
        print("common topics: ", common_topics)
        print("")
    
    # sort messages by timestamp
    messages.sort(key=get_min_date)

    # save messages with timestamps
    with open("new_message_list.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
