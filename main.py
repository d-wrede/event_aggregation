import json
# from ruamel.yaml import YAML
# yaml = YAML(typ='safe')
# import yaml
from copy import copy
#import os
#from private.extract_timestamp import extract_timestamp, filter_string
# from src.organize_timestamps import (
#     dateparser_vs_ownparser,
#     interpret_dates,
#     get_min_date,
# )
from src.extract_topic import (
    extract_topic,
    evaluate_topic_extraction,
)
from optimize_parameters import load_word_freq_dict



json_filename = "chosen_topics.json"

number_of_messages = 600
first_letters = 500
optimization_switch = False

# read messages from json file
# with open(json_filename, "r", encoding="utf-8") as f:
#     messages = json.load(f)

def back_to_floats(parameters):
    # Convert variables to floats and integers
    parameters['keyword_selection_parameters']['spacy_keywords_weight'] = float(parameters['keyword_selection_parameters']['spacy_keywords_weight'])
    parameters['keyword_selection_parameters']['rake_keywords_weight'] = float(parameters['keyword_selection_parameters']['rake_keywords_weight'])
    parameters['keyword_selection_parameters']['tf_IDF_keywords_weight'] = float(parameters['keyword_selection_parameters']['tf_IDF_keywords_weight'])
    parameters['keyword_selection_parameters']['LDA_keywords_weight'] = float(parameters['keyword_selection_parameters']['LDA_keywords_weight'])
    parameters['keyword_selection_parameters']['NMF_keywords_weight'] = float(parameters['keyword_selection_parameters']['NMF_keywords_weight'])
    parameters['keyword_selection_parameters']['frequency_threshold1'] = float(parameters['keyword_selection_parameters']['frequency_threshold1'])
    parameters['keyword_selection_parameters']['frequency_threshold2'] = float(parameters['keyword_selection_parameters']['frequency_threshold2'])
    parameters['keyword_selection_parameters']['frequency_weight1'] = float(parameters['keyword_selection_parameters']['frequency_weight1'])
    parameters['keyword_selection_parameters']['frequency_weight2'] = float(parameters['keyword_selection_parameters']['frequency_weight2'])
    parameters['keyword_selection_parameters']['digit_weight'] = float(parameters['keyword_selection_parameters']['digit_weight'])
    parameters['keyword_selection_parameters']['highest_rank'] = int(parameters['keyword_selection_parameters']['highest_rank'])
    parameters['keyword_selection_parameters']['rank_weight'] = float(parameters['keyword_selection_parameters']['rank_weight'])
    parameters['keyword_selection_parameters']['position_weight'] = float(parameters['keyword_selection_parameters']['position_weight'])
    parameters['keyword_selection_parameters']['position_ratio_weight'] = float(parameters['keyword_selection_parameters']['position_ratio_weight'])
    parameters['spacy']['LOC'] = int(parameters['spacy']['LOC'])
    parameters['spacy']['ORG'] = int(parameters['spacy']['ORG'])
    parameters['spacy']['MISC'] = int(parameters['spacy']['MISC'])
    parameters['rake']['max_length'] = int(parameters['rake']['max_length'])
    parameters['tf_IDF']['min_keywords'] = int(parameters['tf_IDF']['min_keywords'])
    parameters['tf_IDF']['max_keywords'] = int(parameters['tf_IDF']['max_keywords'])
    parameters['tf_IDF']['keywords_multiplier'] = float(parameters['tf_IDF']['keywords_multiplier'])
    parameters['LDA']['num_topics_multiplier'] = float(parameters['LDA']['num_topics_multiplier'])
    parameters['LDA']['passes'] = int(parameters['LDA']['passes'])
    parameters['NMF']['num_topics_multiplier'] = float(parameters['NMF']['num_topics_multiplier'])




def process_messages(word_freq_dict, parameters, messages):
    # Get the directory path of the current file
    # current_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the config.json file
    # config_path = os.path.join(current_directory, 'config', 'params.yaml')
    config_path = "config/params.yaml"

    # Load the configuration file
    # with open(config_path, "r") as file:
    #     parameters = yaml.load(file, Loader=yaml.FullLoader)
    # back_to_floats(parameters)

    # delete in each message the keys "spacy_NER", "rake_keywords", "tf_IDF", "LDA", "NMF", "common_topics" but leave "chosen_topics" in 'topic_suggestions'
    for message in messages:
        if "topic_suggestions" in message:
            message["topic_suggestions"] = {
                "chosen_topics": message["topic_suggestions"]["chosen_topics"]
            }
    

    # count_dates = 0
    # count_messages = 0

    # # timestamp extraction
    # for message in messages[:number_of_messages]:
    #     # and 'approved' not in message['timestamps']['comment']:
    #     if "message" not in message or message["message"] == "":
    #         continue

    #     count_messages += 1

    #     # extract timestamps as list of dicts with date and time using own parser
    #     time_matches = extract_timestamp(message["message"])

    #     # interpret dates by connecting date and time
    #     interpreted_dates = interpret_dates(time_matches)

    #     if len(interpreted_dates):
    #         # add timestamps to message dict
    #         message.setdefault("timestamps", [])
    #         message["timestamps"] = interpreted_dates
    #         count_dates += 1

    # print(f"found {count_dates} dates in {len(messages)} messages")

    # # sort messages by timestamp
    # messages.sort(key=get_min_date)

    # list messages with timestamps and message text
    # filtered_messages = [
    # message
    # for message in messages[:number_of_messages]
    # if 'timestamps' in message
    # ]
    filtered_messages = messages #[:number_of_messages]
    ### extract topic ###
    extract_topic(filtered_messages, copy(parameters), word_freq_dict)

    # filtered_messages_with_selected_keys = [
    # {key: message[key] for key in ('message', 'topic_suggestions', 'timestamps')}
    # for message in filtered_messages
    # ] # TODO: prevent duplicates

    # topic extraction algorithm evaluation
    performance = evaluate_topic_extraction(filtered_messages)
    #print("performance: ", performance)


    # safe in readable format
    # if not optimization_switch:
    #     with open("file.txt", "w", encoding="utf-8") as f:
    #         for message in filtered_messages:
    #             f.write(
    #                 f'### message ###\n{filter_string(message["message"])[:500]} \n---\n'
    #             )
    #             timestamps = message["timestamps"]
    #             for stamp in timestamps:
    #                 timestamp = ""
    #                 if stamp["date1"]:
    #                     timestamp += stamp["date1"]
    #                 if stamp["clock1"]:
    #                     timestamp += " " + stamp["clock1"]
    #                 if stamp["date2"]:
    #                     timestamp += " - " + stamp["date2"]
    #                 if stamp["clock2"]:
    #                     timestamp += " " + stamp["clock2"]

    #                 f.write(f"timestamp: {timestamp} \n")
    #             # topics
    #             f.write(f'topics: {message["topic_suggestions"]["common_topics"][:6]}\n\n')

        # save messages with timestamps
        # with open("new_message_list.json", "w", encoding="utf-8") as f:
        #     json.dump(filtered_messages_with_selected_keys, f, indent=4, ensure_ascii=False)
    
    return performance


if __name__ == "__main__":

    # preloading the word frequency dictionary
    word_freq_dict = load_word_freq_dict()

    process_messages(word_freq_dict, parameters, messages)
