from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from collections import defaultdict
import re
import spacy
import json

# Load spaCy's German model
nlp = spacy.load("de_core_news_lg")

from rake_nltk import Rake

r = Rake(language="german")
import pandas as pd
import nltk
from src.extract_timestamp import filter_string


def spacy_ner(message):
    doc = nlp(message)
    place = []
    topic = []
    misc = []
    for ent in doc.ents:
        if ent.label_ == "LOC":
            place.append(ent.text)
        if ent.label_ == "ORG":
            topic.append(ent.text)
        if ent.label_ == "MISC":
            misc.append(ent.text)

    return place, topic, misc


def rake(message, max_length=3):
    r.extract_keywords_from_text(message)
    scored_keywords = r.get_ranked_phrases_with_scores()

    # Filter out keywords longer than the max_length
    filtered_keywords = [
        kw for score, kw in scored_keywords if len(kw.split()) <= max_length
    ]

    # Find the original case of the keywords in the message
    original_case_keywords = []
    for keyword in filtered_keywords:
        # Escape any special characters in the keyword for regex search
        escaped_keyword = re.escape(keyword)
        # Search for the keyword in the message, case insensitive
        match = re.search(escaped_keyword, message, flags=re.IGNORECASE)
        if match:
            # Append the matched original keyword to the list
            original_case_keywords.append(match.group())

    return original_case_keywords


def cluster_messages(cleaned_texts):
    # Load the pre-trained model
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Generate embeddings for your texts
    embeddings = model.encode(cleaned_texts)

    # Cluster the embeddings using K-means
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Print the cluster assignment for each text
    for text, label in zip(cleaned_texts, cluster_labels):
        print(f"Text: {text}\nCluster: {label}\n")


def tf_IDF(cleaned_texts):
    # Create the TfidfVectorizer with a custom tokenizer and analyzer
    vectorizer = TfidfVectorizer(tokenizer=lambda text: text.split(), analyzer="word")

    # Calculate the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Get the feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Number of top keywords to extract from each text
    num_keywords = 10
    keywords = []

    for index, text in enumerate(cleaned_texts):
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
            match = re.search(
                escaped_keyword, cleaned_texts[index], flags=re.IGNORECASE
            )
            if match:
                # Append the matched original keyword to the list
                original_case_keywords.append(match.group())

        keywords.append(original_case_keywords)

    return keywords


def LDA_topic_modeling(cleaned_texts):
    # Create a list of lists containing words for each text
    texts = [text.split() for text in cleaned_texts]

    # Create a dictionary of words and their corresponding integer ids
    dictionary = corpora.Dictionary(texts)

    # Create the corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model
    num_topics = len(cleaned_texts)
    model = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15
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


def LDA_topic_modeling2(cleaned_texts):
    # Create a list of lists containing words for each text
    texts = [text.split() for text in cleaned_texts]

    # Create a dictionary of words and their corresponding integer ids
    dictionary = corpora.Dictionary(texts)

    # Create the corpus: Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train the LDA model
    num_topics = len(cleaned_texts)
    model = LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15
    )

    # Get the topic distribution for each message
    topic_distributions = model.get_document_topics(corpus)

    # Find the topic with the highest probability for each message
    most_probable_topics = [
        max(topic_dist, key=lambda x: x[1])[0] for topic_dist in topic_distributions
    ]

    # Get the topic keywords
    topic_keywords = model.print_topics(num_words=5)

    # Create a dictionary mapping topic number to its keywords
    topic_keywords_dict = {topic[0]: topic[1] for topic in topic_keywords}

    # Extract the keywords for the most probable topic for each message
    lda_keywords = []
    for topic_num in most_probable_topics:
        keywords_list = [
            keyword_prob.split("*")[1].strip('" ')
            for keyword_prob in topic_keywords_dict[topic_num].split(" + ")
        ]
        lda_keywords.append(keywords_list)

    return lda_keywords


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


def NMF_topic_modeling(cleaned_texts):
    # Define the number of topics you want to extract
    n_topics = len(cleaned_texts)

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Create the NMF model and fit it to the TF-IDF matrix
    nmf = NMF(n_components=n_topics, random_state=42)
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


def find_common_topics(keyword_dicts, text):
    term_count = defaultdict(float)
    longest_terms = {}
    text_length = len(text)
    weights = {
        "spacy_NER": 449,
        "rake_keywords": 580,
        "tf_IDF": 1264,
        "LDA": 250,
        "NMF": 829,
        "common_topics": 1891,
    }
    # using a frequency dictionary to store the frequency of each word due to timing issues
    frequency_dict = {}
    word_freq_dict = load_word_freq_dict()

    # Exclude 'common_topics' from the calculation
    weights_without_common_topics = {
        k: v for k, v in weights.items() if k != "common_topics"
    }

    # Calculate the average
    average_weights = sum(weights_without_common_topics.values()) / len(
        weights_without_common_topics
    )

    for algorithm, keywords in keyword_dicts.items():
        algorithm_weight = (weights[algorithm] / average_weights) * 20
        # ensure to only search for keywords that are not already in the frequency dictionary
        new_keywords = [key for key in keywords if key not in frequency_dict]
        frequency_dict.update(word_frequency(new_keywords, word_freq_dict))

        for rank, keyword in enumerate(keywords):
            # Find the index position of the keyword in the text
            index_position = text.find(keyword)

            if index_position != -1:
                # Calculate the position-based weight
                position_weight = (1 - (index_position / text_length) * 1.5) * 100
                highest = 35
                rankweight = (highest - rank if rank < highest else 1) * 2

                # Calculate the frequency-based weight
                frequency_threshold1 = 0.6 * 10**6
                frequency_threshold2 = 1.2 * 10**6
                if frequency_dict[keyword] > frequency_threshold2:
                    frequency_weight = -150
                elif frequency_dict[keyword] > frequency_threshold1:
                    frequency_weight = -75
                else:
                    frequency_weight = 0

                # Assign a score based on the order of the keyword (higher rank = lower score) and position weight
                score = (
                    rankweight + algorithm_weight + position_weight + frequency_weight
                )
                if keyword in [
                    "März",
                    "Berührung",
                    "tänzerisch",
                    "Muskeln",
                    "gesucht",
                    "Falls",
                    "com",
                    "Freude",
                    "kopiert",
                ]:
                    print(
                        f"keyword: {keyword}\nrankweight: {rankweight}\nalgorithm_weight: {algorithm_weight}\nposition_weight: {position_weight}\nfrequency: {frequency_dict[keyword]}\nfrequency_weight: {frequency_weight}\nscore: {score}\n\n"
                    )
                    pass
                    # print(f"score: {score} for {keyword}")
            else:
                score = 7 - rank if rank < 7 else 1

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

    print("term_count: ", term_count)
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


def remove_stopwords(text):
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Remove stop words and create a list of filtered tokens
    filtered_tokens = [token.text for token in doc if not token.is_stop]

    return " ".join(filtered_tokens)


def check_thema(message):
    """catch the topic by 'Thema:' in message, if available"""

    if "Thema:" in message["message"]:
        index_position = message["message"].find("Thema:") + len("Thema:")
        newline_position = (
            message["message"][index_position:].find("\n") + index_position
        )
        if newline_position == -1:
            newline_position = len(message["message"])
        topic = message["message"][index_position:newline_position].strip()
        print("the topic is: ", topic)
        return topic
    else:
        return None


def get_thema_topic(message):
    if "Thema:" in message["message"]:
        index_position = message["message"].find("Thema:") + len("Thema:")
        newline_position = (
            message["message"][index_position:].find("\n") + index_position
        )
        if newline_position == -1:
            newline_position = len(message["message"])
        return message["message"][index_position:newline_position].strip()
    return None


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


def filter_keywords(keywords):
    filtered_keywords = []
    lowercase_keywords = set()

    for keyword in keywords:
        # Create a spaCy token from the keyword
        token = nlp(keyword)[0]

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
            and not token.is_stop
            and valid_digit_rule
            and not contains_link
            and not_duplicate
        ):
            filtered_keywords.append(keyword)
            lowercase_keywords.add(keyword.lower())

    return filtered_keywords


# Filter the DataFrame to only include rows with a frequency above 600,000
# filtered_df = df[df['freq'] > 600000]

# # Save the filtered DataFrame to a new CSV file
# filtered_df.to_csv('filtered_decow_wordfreq_cistem.csv')
# print("saved filtered df")

# # Calculate the number of words in the filtered DataFrame and the number of words that have been filtered out
# num_words_filtered = len(filtered_df)
# num_words_filtered_out = len(df) - num_words_filtered

# print(f"Number of words in the filtered DataFrame: {num_words_filtered}")
# print(f"Number of words filtered out: {num_words_filtered_out}")


def load_word_freq_dict():
    """Load the word frequency dictionary"""
    df = pd.read_csv("high_frequency_decow_wordfreq_cistem.csv", index_col=["word"])
    return df["freq"].to_dict()


def word_frequency(word_list, word_freq_dict):
    stemmer = nltk.stem.Cistem()

    # Define a function to categorize German words based on frequency
    def german_word_frequency(word):
        try:
            stemmed_word = stemmer.stem(word.lower())
            freq = word_freq_dict.get(stemmed_word)
        except Exception as e:
            print(f"Error stemming the word '{word}': {e}")
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

    entry_frequencies = {entry: process_entry(entry) for entry in word_list}

    return entry_frequencies


def extract_keywords(cleaned_texts):
    # spaCy NER and RAKE
    spacy_keywords = []
    rake_keywords = []
    for cleaned_message in cleaned_texts:
        place, topic, misc = spacy_ner(cleaned_message)
        spacy_keywords.append(place + topic + misc)
        rake_keywords.append(rake(cleaned_message))

    # TF-IDF, LDA, NMF
    tf_IDF_keywords = tf_IDF(cleaned_texts)
    LDA_keywords = LDA_topic_modeling(cleaned_texts)
    LDA_keywords = sort_keywords_by_input_order(LDA_keywords, cleaned_texts)
    NMF_keywords = NMF_topic_modeling(cleaned_texts)
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
):
    for i, (message_idx, _) in enumerate(cleaned_texts_with_indices):
        message = filtered_messages[message_idx]
        message["topic_suggestions"]["spacy_NER"] = filter_keywords(spacy_keywords[i])
        message["topic_suggestions"]["rake_keywords"] = filter_keywords(
            rake_keywords[i]
        )
        message["topic_suggestions"]["tf_IDF"] = filter_keywords(tf_IDF_keywords[i])
        message["topic_suggestions"]["LDA"] = filter_keywords(LDA_keywords[i])
        message["topic_suggestions"]["NMF"] = filter_keywords(NMF_keywords[i])


def extract_common_topics(filtered_messages, first_letters):
    """Extract common topics, using the function 'find_common_topics',
    from the messages and store them in the message dictionaries."""
    for message in filtered_messages:
        if "common_topics" in message["topic_suggestions"]:
            continue

        print("message: ", filter_string(message["message"][:first_letters]))

        common_topics = find_common_topics(
            message["topic_suggestions"],
            filter_string(message["message"][:first_letters]),
        )
        common_topics = filter_keywords(common_topics)
        message["topic_suggestions"]["common_topics"] = common_topics

        print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])
        print("rake_keywords: ", message["topic_suggestions"]["rake_keywords"])
        print("## later added ##")
        print("tf_IDF: ", message["topic_suggestions"]["tf_IDF"])
        print("LDA: ", message["topic_suggestions"]["LDA"])
        print("NMF: ", message["topic_suggestions"]["NMF"])

        # print timestamps
        timestamps = message["timestamps"]
        for stamp in timestamps:
            if stamp["date1"]:
                print("date1: ", stamp["date1"])
            if stamp["clock1"]:
                print("clock1: ", stamp["clock1"])
            if stamp["date2"]:
                print("date2: ", stamp["date2"])
            if stamp["clock2"]:
                print("clock2: ", stamp["clock2"])
            print("---")

        print("common topics: ", common_topics)
        print("")


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
    with open("topics.json", "r", encoding="utf-8") as f:
        evaluated_messages = json.load(f)

    # step through each message and find the score
    for message in filtered_messages:
        # find the respective evaluated message
        found_message = False
        for evaluated_message in evaluated_messages:
            if evaluated_message["id"] == message["id"]:
                # get selected topics
                evaluated_topics = evaluated_message["common_topics"]
                found_message = True

        if not found_message:
            print("message not found: ", message["id"])
            continue

        # compare the common topics in 'topics' with the common topics in 'message'
        # step through each list in the dictionary
        for key, topics in message["topic_suggestions"].items():
            # step through each topic in the list
            for i, topic in enumerate(topics):
                # step through each selected topic
                for j, evaluated_topic in enumerate(evaluated_topics):
                    if topic == evaluated_topic:
                        score = 30 - i - j
                        performance[key] += score
                        if score < 0:
                            print("score < 0: ", score)
    print("performance: ", performance)


def extract_topic(filtered_messages, first_letters):
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

    # get keywords for each message
    (
        spacy_keywords,
        rake_keywords,
        tf_IDF_keywords,
        LDA_keywords,
        NMF_keywords,
    ) = extract_keywords([text for _, text in cleaned_texts_with_indices])
    store_keywords_in_messages(
        filtered_messages,
        spacy_keywords,
        rake_keywords,
        tf_IDF_keywords,
        LDA_keywords,
        NMF_keywords,
        cleaned_texts_with_indices,
    )
    extract_common_topics(filtered_messages, first_letters)
