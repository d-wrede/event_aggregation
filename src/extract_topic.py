# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# from gensim.parsing.preprocessing import (
#     preprocess_string,
#     strip_punctuation,
#     strip_numeric,
# )
from collections import defaultdict
import re
import spacy
import json
import warnings

# Load spaCy's German model
# nlp = spacy.load("de_core_news_lg")

from rake_nltk import Rake

r = Rake(language="german")
import pandas as pd
import nltk
from src.extract_timestamp import filter_string


def spacy_ner(messages, parameters, nlp):
    """Extracts named entities from the messages using spaCy's NER model."""
    # Calculate the average number of words in the messages
    avg_words = sum(len(message.split()) for message in messages) / len(messages)
    # Calculate the number of messages, which are processed in one batch
    batch_size = round(avg_words * parameters["batch_size"])

    results = []
    # Process the messages in batches and extract
    # locations, organizations and miscellaneous entities
    for doc in nlp.pipe(messages, batch_size=batch_size):
        keywords = []
        for ent in doc.ents:
            if ent.label_ in ("LOC", "ORG", "MISC") and parameters.get(ent.label_):
                keywords.append(ent.text)
        results.append(keywords)
    return results


def rake(messages, parameters, stopwords):
    """Extracts keywords from the messages using the RAKE algorithm."""

    # Initialize RAKE with stopword list and length filters
    r = Rake(
        stopwords=stopwords,
        min_length=parameters["min_length"],
        max_length=parameters["max_length"],
    )

    results = []

    for message in messages:
        r.extract_keywords_from_text(message)
        scored_keywords = r.get_ranked_phrases_with_scores()

        # Filter out keywords longer than the max_length and find the original case
        original_case_keywords = [
            match.group()
            for score, kw in scored_keywords
            for match in [re.search(re.escape(kw), message, flags=re.IGNORECASE)]
            if match
        ]
        results.append(original_case_keywords)

    return results


# def cluster_messages(cleaned_texts, model):
#     # Load the pre-trained model
#     model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

#     # Generate embeddings for your texts
#     embeddings = model.encode(cleaned_texts)

#     # Cluster the embeddings using K-means
#     num_clusters = 3
#     kmeans = KMeans(n_clusters=num_clusters)
#     cluster_labels = kmeans.fit_predict(embeddings)

#     # Print the cluster assignment for each text
#     # for text, label in zip(cleaned_texts, cluster_labels):
#     #     print(f"Text: {text}\nCluster: {label}\n")


def tf_IDF(cleaned_texts, parameters):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None'",
        )

        # Create the TfidfVectorizer with a custom tokenizer, analyzer, and the new parameters
        vectorizer = TfidfVectorizer(
            tokenizer=lambda text: text.split(),
            analyzer="word",
            max_df=parameters["max_df"],
            min_df=parameters["min_df"],
            ngram_range= (parameters["ngram_range1"], parameters["ngram_range2"]),
            max_features=parameters["max_features"],
        )

        # Calculate the TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

        # Get the feature names (words)
        feature_names = vectorizer.get_feature_names_out()

    min_keywords = parameters["min_keywords"]
    max_keywords = parameters["max_keywords"]

    keywords = []

    for index, text in enumerate(cleaned_texts):
        # Number of top keywords to extract from each text
        num_keywords = int(len(text.split()) * parameters["keywords_multiplier"])

        # Clip the value to the defined boundaries
        num_keywords = int(max(min_keywords, min(num_keywords, max_keywords)))

        # Get the indices of the top num_keywords features in the text
        top_feature_indices = (
            tfidf_matrix[index].toarray()[0].argsort()[-num_keywords:][::-1]
        )

        # Get the top keywords and their corresponding TF-IDF scores
        top_keywords_and_scores = [
            (feature_names[i], tfidf_matrix[index, i]) for i in top_feature_indices
        ]

        original_case_keywords = []
        for keyword, score in top_keywords_and_scores:
            # Escape any special characters in the keyword for regex search
            escaped_keyword = re.escape(keyword)
            # Search for the keyword in the message, case insensitive
            if match := re.search(
                escaped_keyword, cleaned_texts[index], flags=re.IGNORECASE
            ):
                # Append the matched original keyword to the list
                original_case_keywords.append(match.group())
        original_case_keywords = [
            keyword.replace(",", " ").strip() for keyword in original_case_keywords
        ]
        keywords.append(original_case_keywords)
    return keywords


def LDA_topic_modeling(cleaned_texts, parameters):
    # Create a list of lists containing words for each text
    texts = [text.split() for text in cleaned_texts]

    # Create a dictionary of words and their corresponding integer ids
    dictionary = corpora.Dictionary(texts)

    # Create the corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model
    num_topics = int(len(cleaned_texts) * parameters["num_topics_multiplier"])
    if num_topics == 0:
        num_topics = 1
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=parameters["passes"],
    )

    # Parse the topics to get lists of keywords
    parsed_topics = []
    topics = model.print_topics(num_words=5)
    for topic in topics:
        topic_keywords = topic[1]
        keywords_list = [
            keyword_prob.split("*")[1].strip('" ')
            for keyword_prob in topic_keywords.split(" + ")
        ]
        parsed_topics.append(keywords_list)

    return parsed_topics


def NMF_topic_modeling(cleaned_texts, parameters):
    """Extract topics from a list of cleaned texts using NMF."""
    # TODO: consider adding more parameters
    # init: This parameter controls the initialization method for the NMF algorithm. You can experiment with different initialization methods such as 'random', 'nndsvd', 'nndsvda', and 'nndsvdar' to see if they lead to better performance.
    # solver: You can try both 'cd' (Coordinate Descent) and 'mu' (Multiplicative Update) solvers to see which one works better for your specific problem.
    # beta_loss: This parameter is used only in the 'mu' solver and represents the type of beta divergence to minimize. You can experiment with 'frobenius', 'kullback-leibler', and 'itakura-saito' to see if they improve the algorithm's performance.
    # tol: This parameter controls the tolerance for the stopping condition. You can try different values to see if they lead to better performance.




    # Define the number of topics you want to extract
    num_topics = int(len(cleaned_texts) * parameters["num_topics_multiplier"])
    if num_topics == 0:
        num_topics = 1

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Create the NMF model and fit it to the TF-IDF matrix
    nmf = NMF(
        n_components=num_topics,
        max_iter=parameters["max_iter"],
        tol=parameters["tol"],
        alpha_W=parameters["alpha_W"],
        alpha_H=parameters["alpha_H"],
        l1_ratio=parameters["l1_ratio"],
        random_state=42
    )

    nmf.fit(tfidf_matrix)

    # Extract topics and associated words
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        original_case_keywords = []

        for keyword in top_words:
            # Search for the keyword in the cleaned_texts, case insensitive
            for i, cleaned_text in enumerate(cleaned_texts):
                escaped_keyword = re.escape(keyword)
                match = re.search(escaped_keyword, cleaned_text, flags=re.IGNORECASE)
                if match:
                    original_case_keywords.append(match.group())
                    break

        topics.append(original_case_keywords)

    return topics


def sort_keywords_by_input_order(lda_keywords, cleaned_texts):
    sorted_lda_keywords = [[] for _ in range(len(cleaned_texts))]
    used_keywords = set()

    for topic_keywords in lda_keywords:
        for keyword in topic_keywords:
            if keyword not in used_keywords:
                for i, cleaned_text in enumerate(cleaned_texts):
                    if keyword in cleaned_text:
                        sorted_lda_keywords[i].append(keyword)
                        used_keywords.add(keyword)
                        break

    return sorted_lda_keywords


def find_common_topics(keyword_dicts, text, parameters, word_freq_dict):
    term_count = defaultdict(float)
    longest_terms = {}
    text_length = len(text)
    weights = {
        "spacy_NER": parameters["spacy_keywords_weight"],
        "rake_keywords": parameters["rake_keywords_weight"],
        "tf_IDF": parameters["tf_IDF_keywords_weight"],
        "LDA": parameters["LDA_keywords_weight"],
        "NMF": parameters["NMF_keywords_weight"],
    }
    # using a frequency dictionary to store the frequency of each word due to timing issues
    frequency_dict = {}
    # word_freq_dict = load_word_freq_dict()

    for algorithm, keywords in keyword_dicts.items():
        # avoid the manually chosen topics being used in the common topics
        if algorithm == "chosen_topics":
            continue
        algorithm_weight = weights[algorithm]
        # ensure to only search for keywords that are not already in the frequency dictionary
        new_keywords = [key for key in keywords if key and key not in frequency_dict]
        frequency_dict.update(word_frequency(new_keywords, word_freq_dict))

        for rank, keyword in enumerate(keywords):
            # Skip empty keywords
            if not keyword:
                continue
            # Calculate the weight based on the rank of the keyword
            highest = parameters["highest_rank"]
            rankweight = (highest - rank if rank < highest else 1) * parameters[
                "rank_weight"
            ]

            # Calculate the frequency-based weight
            frequency_threshold1 = parameters["frequency_threshold1"] * 10**6
            frequency_threshold2 = parameters["frequency_threshold2"] * 10**6
            if frequency_dict[keyword] > frequency_threshold2:
                frequency_weight = parameters["frequency_weight2"]
            elif frequency_dict[keyword] > frequency_threshold1:
                frequency_weight = parameters["frequency_weight1"]
            else:
                frequency_weight = 0

            digit_weight = (
                sum(char.isdigit() for char in keyword)
                / len(keyword)
                * parameters["digit_weight"]
            )

            # Find the index position of the keyword in the text
            index_position = text.find(keyword)

            if index_position != -1:
                # Calculate the position-based weight
                position_weight = (
                    1
                    - (index_position / text_length)
                    * parameters["position_ratio_weight"]
                ) * parameters[
                    "position_weight"
                ]  # TODO: describe as a function

                # Assign a score based on the calculated weights
                score = (
                    rankweight
                    + algorithm_weight
                    + position_weight
                    + frequency_weight
                    + digit_weight
                )
            else:
                # calculate the score without the position weight
                score = rankweight + algorithm_weight + frequency_weight + digit_weight

            term_count[keyword] += score

            # If the keyword is a substring of a longer term, update the longest_terms dictionary
            if keyword in longest_terms:
                if (
                    len(longest_terms[keyword]) < len(keyword)
                    and len(keyword.split()) <= 2
                ):
                    longest_terms[keyword] = keyword
            elif len(keyword.split()) <= 2:
                longest_terms[keyword] = keyword

    # Sort the terms by their score in descending order
    sorted_terms = sorted(term_count.items(), key=lambda x: x[1], reverse=True)

    # Create a list of the most common terms, using the longest form of the term
    most_common_terms = []
    for term, score in sorted_terms:
        if term in longest_terms:
            is_subset = False
            for i, common_term in enumerate(most_common_terms):
                if term in common_term:
                    is_subset = True
                    break
                elif common_term in term:
                    most_common_terms[i] = term
                    is_subset = True
                    break

            if not is_subset:
                most_common_terms.append(longest_terms[term])

    return most_common_terms


def remove_stopwords(text, nlp):
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Remove stop words and create a list of filtered tokens
    filtered_tokens = [token.text for token in doc if not token.is_stop]

    return " ".join(filtered_tokens)


# def check_thema(message):
#     """catch the topic by 'Thema:' in message, if available"""

#     if "Thema:" in message["message"]:
#         index_position = message["message"].find("Thema:") + len("Thema:")
#         newline_position = (
#             message["message"][index_position:].find("\n") + index_position
#         )
#         if newline_position == -1:
#             newline_position = len(message["message"])
#         topic = message["message"][index_position:newline_position].strip()
#         return topic
#     else:
#         return None


# def get_thema_topic(message):
#     if "Thema:" in message["message"]:
#         index_position = message["message"].find("Thema:") + len("Thema:")
#         newline_position = (
#             message["message"][index_position:].find("\n") + index_position
#         )
#         if newline_position == -1:
#             newline_position = len(message["message"])
#         return message["message"][index_position:newline_position].strip()
#     return None


def check_if_topic(filtered_messages):
    for message in filtered_messages:
        if "Thema:" in message["message"]:
            index_position = message["message"].find("Thema:") + len("Thema:")
            newline_position = (
                message["message"][index_position:].find("\n") + index_position
            )
            if newline_position == -1:
                newline_position = len(message["message"])
            topic = message["message"][index_position:newline_position].strip()
            message["topic_suggestions"]["common_topics"] = topic


def filter_keywords(keywords, nlp):
    filtered_keywords = []
    lowercase_keywords = set()

    # only keywords with min length of 3
    keywords = [keyword for keyword in keywords if len(keyword) > 2]
    # Create spaCy tokens from the keywords
    tokens = [doc[0] for doc in nlp.pipe(keywords)]

    for i, keyword in enumerate(keywords):
        # Find all sets of digits in the keyword
        digit_sets = re.findall(r"\d+", keyword)

        # Check if the keyword has only one set of digits with a maximum of 3 digits
        valid_digit_rule = len(digit_sets) <= 1 and all(
            len(ds) <= 3 for ds in digit_sets
        )

        # Check if the keyword contains a link
        contains_link = "http" in keyword.lower() or "www" in keyword.lower()

        # Check if the keyword is not already in the set
        not_duplicate = keyword.lower() not in lowercase_keywords

        if (
            len(keyword) > 2
            and not tokens[i].is_stop
            and valid_digit_rule
            and not contains_link
            and not_duplicate
        ):
            filtered_keywords.append(keyword)
            lowercase_keywords.add(keyword.lower())

    return filtered_keywords


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency04_decow_wordfreq_cistem.csv", index_col=["word"])
    print("Word frequency dictionary loaded in extract_topic.py")
    return df["freq"].to_dict()


def word_frequency(word_list, word_freq_dict):
    stemmer = nltk.stem.Cistem()

    # Define a function to categorize German words based on frequency
    def german_word_frequency(word):
        try:
            stemmed_word = stemmer.stem(word.lower())
            freq = word_freq_dict.get(stemmed_word)
        except Exception as e:
            freq = None

        if freq is None:
            freq = word_freq_dict.get(word.lower(), 0)
        return freq

    # Process each entry in the word_list
    def process_entry(entry):
        words = entry.split()
        word_frequencies = [german_word_frequency(word) for word in words]
        # Use the lowest frequency of individual words
        combined_frequency = min(word_frequencies)
        return combined_frequency

    entry_frequencies = {entry: process_entry(entry) for entry in word_list if entry}
    return entry_frequencies


def extract_keywords(cleaned_texts, parameters, nlp, stopwords):
    """Extract keywords from the cleaned texts"""
    rake_keywords = rake(cleaned_texts, parameters["rake"], stopwords)
    spacy_keywords = spacy_ner(cleaned_texts, parameters["spacy"], nlp)
    tf_IDF_keywords = tf_IDF(cleaned_texts, parameters["tf_IDF"])
    LDA_keywords = LDA_topic_modeling(cleaned_texts, parameters["LDA"])
    LDA_keywords = sort_keywords_by_input_order(LDA_keywords, cleaned_texts)
    NMF_keywords = NMF_topic_modeling(cleaned_texts, parameters["NMF"])
    NMF_keywords = sort_keywords_by_input_order(NMF_keywords, cleaned_texts)

    return spacy_keywords, rake_keywords, tf_IDF_keywords, LDA_keywords, NMF_keywords


def store_keywords_in_messages(
    filtered_messages,
    spacy_keywords,
    rake_keywords,
    tf_IDF_keywords,
    LDA_keywords,
    NMF_keywords,
    cleaned_texts_with_indices,
    nlp_spacy,
):
    for i, (message_idx, _) in enumerate(cleaned_texts_with_indices):
        message = filtered_messages[message_idx]
        message["topic_suggestions"]["spacy_NER"] = spacy_keywords[i]
        message["topic_suggestions"]["rake_keywords"] = rake_keywords[i]
        message["topic_suggestions"]["tf_IDF"] = tf_IDF_keywords[i]
        message["topic_suggestions"]["LDA"] = LDA_keywords[i]
        message["topic_suggestions"]["NMF"] = NMF_keywords[i]


def extract_common_topics(
    filtered_messages, first_letters, parameters, word_freq_dict, nlp_spacy
):
    """Extract common topics, using the function 'find_common_topics',
    from the messages and store them in the message dictionaries."""
    for message in filtered_messages:
        if "common_topics" in message["topic_suggestions"]:
            continue

        # print("message: ", filter_string(message["message"][:first_letters]))

        common_topics = find_common_topics(
            message["topic_suggestions"],
            filter_string(message["message"][:first_letters]),
            parameters["keyword_selection_parameters"],
            word_freq_dict,
        )
        common_topics = filter_keywords(common_topics, nlp_spacy)
        message["topic_suggestions"]["common_topics"] = common_topics

        # print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])
        # print("rake_keywords: ", message["topic_suggestions"]["rake_keywords"])
        # print("## later added ##")
        # print("tf_IDF: ", message["topic_suggestions"]["tf_IDF"])
        # print("LDA: ", message["topic_suggestions"]["LDA"])
        # print("NMF: ", message["topic_suggestions"]["NMF"])

        # print timestamps
        # timestamps = message["timestamps"]
        # for stamp in timestamps:
        #     if stamp["date1"]:
        #         print("date1: ", stamp["date1"])
        #     if stamp["clock1"]:
        #         print("clock1: ", stamp["clock1"])
        #     if stamp["date2"]:
        #         print("date2: ", stamp["date2"])
        #     if stamp["clock2"]:
        #         print("clock2: ", stamp["clock2"])
        #     print("---")

        # print("common topics: ", common_topics)
        # print("")


def extract_topic(
    filtered_messages, first_letters, parameters, word_freq_dict, nlp_spacy
):
    # add topic_suggestions key to each message
    for message in filtered_messages:
        message.setdefault("topic_suggestions", {})

    check_if_topic(filtered_messages)

    # Clean and preprocess the texts
    cleaned_texts_with_indices = [
        (idx, filter_string(message["message"][:first_letters]))
        for idx, message in enumerate(filtered_messages)
        if "common_topics" not in message["topic_suggestions"]
    ]

    stopwords = nlp_spacy.Defaults.stop_words
    # get keywords for each message
    (
        spacy_keywords,
        rake_keywords,
        tf_IDF_keywords,
        LDA_keywords,
        NMF_keywords,
    ) = extract_keywords(
        [text for _, text in cleaned_texts_with_indices],
        parameters,
        nlp_spacy,
        stopwords,
    )
    store_keywords_in_messages(
        filtered_messages,
        spacy_keywords,
        rake_keywords,
        tf_IDF_keywords,
        LDA_keywords,
        NMF_keywords,
        cleaned_texts_with_indices,
        nlp_spacy,
    )
    extract_common_topics(
        filtered_messages, first_letters, parameters, word_freq_dict, nlp_spacy
    )


def evaluate_topic_extraction(filtered_messages):
    """compare and score algorithms according to their performance"""
    # create performance dictionary to store the scores
    performance = {
        "spacy_NER": 0,
        "rake_keywords": 0,
        "tf_IDF": 0,
        "LDA": 0,
        "NMF": 0,
        "common_topics": 0,
    }
    # load the evaluated messages
    # with open("topics.json", "r", encoding="utf-8") as f:
    #     evaluated_messages = json.load(f)

    # step through each message and find the score
    for message in filtered_messages:
        # find the respective evaluated message
        # found_message = False
        # for evaluated_message in evaluated_messages:
        #     if evaluated_message["message"][:50] == message["message"][:50]: # TODO: change to message["id"] == evaluated_message["id"]
        #         # get selected topics
        #         evaluated_topics = evaluated_message["common_topics"]
        #         found_message = True

        # if not found_message:
        #     print("message not found: ", message["id"])
        #     continue

        # compare the common topics in 'topics' with the common topics in 'message'
        # step through each list in the dictionary
        for key, topics in message["topic_suggestions"].items():
            if key == "chosen_topics":
                continue
            # step through each topic in the list
            for i, topic in enumerate(topics):
                # step through each selected topic
                for j, chosen_topic in enumerate(
                    message["topic_suggestions"]["chosen_topics"]
                ):
                    if topic == chosen_topic:
                        score = 30 - i - j
                        performance[key] += score
                        # if score < 0:
                        #     print("score < 0: ", score)
    return performance
