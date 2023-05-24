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

# preload spacy model and stopwords for faster processing
nlp_spacy = spacy.load("de_core_news_lg", disable=["parser", "tagger"])
stopwords = nlp_spacy.Defaults.stop_words

import json
import warnings

# Load spaCy's German model
# nlp = spacy.load("de_core_news_lg")

from rake_nltk import Rake
import pandas as pd
import nltk
from src.extract_timestamp import filter_string

# printswitch for debugging, due to multiprocessing
printss = False


def spacy_ner(messages, parameters, spacy_docs):
    """Extracts named entities from the messages using spaCy's NER model.
    input: messages: list of strings
    input: parameters: dictionary
    input: nlp_spacy: spacy model
    output: list of lists of tuples (keyword, score) for each message
    """
    # Calculate the average number of words in the messages
    # avg_words = sum(len(message.split()) for message in messages) / len(messages)
    # Calculate the number of messages, which are processed in one batch
    # uncommented for know
    # batch_size = round(avg_words * parameters["batch_size"])

    results = []
    # Process the messages in batches and extract
    # locations, organizations and miscellaneous entities

    for doc in spacy_docs:
        keywords = []
        for ent in doc.ents:
            # Iterate over parameters in descending order
            if ent.label_ in ("LOC", "ORG", "MISC"):
                # spacy does unfortunately not provide a score for the entities
                # therefore the score is set equal to the parameter value for all entities
                score = parameters[ent.label_]
                keyword_tuple = (ent.text, score)
                keywords.append(keyword_tuple)

        # Sort the keywords based on the score in descending order
        sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        results.append(sorted_keywords)
    return results


def rake(messages, parameters):
    """Extracts keywords from the messages using the RAKE algorithm."""

    # Initialize RAKE with stopword list and length filters
    rake_object = Rake(
        language="german",
        stopwords=stopwords,
        min_length=1,  # parameters["min_length"],
        max_length=parameters["max_length"],
    )

    # rake_object.max_length = parameters["max_length"]
    # min_length=2, max_length=4
    results = []

    for message in messages:
        rake_object.extract_keywords_from_text(message)
        scored_keywords = rake_object.get_ranked_phrases_with_scores()

        # Filter out keywords longer than the max_length and find the original case
        original_case_keywords = [
            (match.group(), score)
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
        
        if printss: 
            print("grab me")
        # ngram range options
        ngram_options = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (1, 4), (2, 4), (3, 4), (4, 4)]
        ngram_range = ngram_options[parameters["ngram_range_index"]]
        if printss:
            print("vectorizer go")
        # Create the TfidfVectorizer with a custom tokenizer, analyzer, and the new parameters
        vectorizer = TfidfVectorizer(
            tokenizer=lambda text: text.split(),
            analyzer="word",
            #max_df=parameters["max_df"],
            #min_df=parameters["min_df"],
            #ngram_range=ngram_range,
            #max_features=parameters["max_features"],
        )

        # Calculate the TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

        # Get the feature names (words)
        feature_names = vectorizer.get_feature_names_out()
    if printss:
        print("grab min keywords")
    # min_keywords = parameters["min_keywords"]
    # max_keywords = parameters["max_keywords"]

    keywords = []

    for index, text in enumerate(cleaned_texts):
        # Number of top keywords to extract from each text
        num_keywords = int(len(text) * 1.8) # * parameters["num_keywords_multiplier"])
        if num_keywords == 0:
            num_keywords = 1

        if printss:
            print("grabbed num keywords")

        # Clip the value to the defined boundaries
        # num_keywords = int(max(min_keywords, min(num_keywords, max_keywords)))

        # Get the indices of the top num_keywords features in the text
        top_feature_indices = (
            tfidf_matrix[index].toarray()[0].argsort()[-num_keywords:][::-1]
        )

        # Get the top keywords and their corresponding TF-IDF scores
        top_keywords_and_scores = [
            (feature_names[i], tfidf_matrix[index, i]) for i in top_feature_indices
        ]

        # top_keywords_and_scores = [
        #     (feature_names[i], tfidf_matrix[index, i]) for i in range(tfidf_matrix.shape[1])
        # ]

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
            (keyword.replace(",", " ").strip(), score)
            for keyword, score in top_keywords_and_scores
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
    num_topics = int(parameters["num_topics_multiplier"])  # len(cleaned_texts))
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
    topics = model.print_topics(num_words=parameters["num_words"])
    for topic in topics:
        topic_keywords = topic[1]
        keywords = [
            (keyword_prob.split("*")[1].strip('" '), float(keyword_prob.split("*")[0]))
            for keyword_prob in topic_keywords.split(" + ")
        ]
        parsed_topics.append(keywords)

    return parsed_topics


def NMF_topic_modeling(cleaned_texts, parameters):
    """Extract topics from a list of cleaned texts using NMF."""
    # TODO: consider adding more parameters
    # init: This parameter controls the initialization method for the NMF algorithm. You can experiment with different initialization methods such as 'random', 'nndsvd', 'nndsvda', and 'nndsvdar' to see if they lead to better performance.
    # solver: You can try both 'cd' (Coordinate Descent) and 'mu' (Multiplicative Update) solvers to see which one works better for your specific problem.
    # beta_loss: This parameter is used only in the 'mu' solver and represents the type of beta divergence to minimize. You can experiment with 'frobenius', 'kullback-leibler', and 'itakura-saito' to see if they improve the algorithm's performance.
    # tol: This parameter controls the tolerance for the stopping condition. You can try different values to see if they lead to better performance.

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # Define the number of topics you want to extract
    num_topics = int(
        tfidf_matrix.shape[0] * 2
    )  # * parameters["num_topics_multiplier"])
    if num_topics == 0:
        num_topics = 1

    # Create the NMF model and fit it to the TF-IDF matrix
    nmf = NMF(
        # consider using multiplicative update solver vs coordinate descent solver
        n_components=num_topics,
        max_iter=parameters["max_iter"],
        tol=parameters["tol"] / 10000,
        alpha_W=parameters["alpha_W"],
        alpha_H=parameters["alpha_H"],
        l1_ratio=parameters["l1_ratio"],
        random_state=42,
    )

    nmf.fit(tfidf_matrix)

    # Extract topics and associated words
    feature_names = vectorizer.get_feature_names_out()
    topics = []

    for topic_idx, topic in enumerate(nmf.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        top_scores = [topic[i] for i in topic.argsort()[:-6:-1]]
        original_case_keywords = []

        for keyword, score in zip(top_words, top_scores):
            for i, cleaned_text in enumerate(cleaned_texts):
                escaped_keyword = re.escape(keyword)
                match = re.search(escaped_keyword, cleaned_text, flags=re.IGNORECASE)
                if match:
                    original_case_keywords.append((match.group(), score))
                    break

        topics.append(original_case_keywords)

    return topics


def sort_keywords(topic_keywords, cleaned_texts):
    """
    Sorts the keywords extracted by a topic modeling algorithm based on the input order of the documents.
    Args:
        topic_keywords (list): A list of lists containing (keyword, probability) tuples for each topic.
        documents (list): A list of documents or texts.
    Returns:
        list: A list of lists containing the sorted keywords for each document along with their probabilities.
    """
    sorted_keywords = [[] for _ in range(len(cleaned_texts))]
    used_keywords = set()

    for keywords in topic_keywords:
        for keyword, probability in keywords:
            if keyword not in used_keywords:
                for i, document in enumerate(cleaned_texts):
                    if keyword in document:
                        sorted_keywords[i].append((keyword, probability))
                        used_keywords.add(keyword)
                        break

    return sorted_keywords


def find_common_topics(keyword_dicts, text, parameters, word_freq_dict):
    """
    Finds the most common topics in a text based on the results of several keyword extraction algorithms.

    The function assigns weights to keywords based on different factors such as their frequency,
    position in the text, and whether they contain digits. It then sums up the weighted scores for
    each keyword across all algorithms, and returns a list of the most common topics, each represented
    by the longest form of the keyword found.

    Args:
        keyword_dicts (dict): A dictionary where each key is the name of a keyword extraction algorithm
                              and the value is a list of tuples, each containing a keyword and its score.
        text (str): The text from which the keywords were extracted.
        parameters (dict): A dictionary of parameters used for weighing the keywords. Includes the
                           weight assigned to each algorithm and various factors used in the calculation
                           of keyword weights.
        word_freq_dict (dict): A dictionary where each key is a word and the value is its frequency
                               in the corpus.

    Returns:
        list: A list of the most common topics, each represented by the longest form of the keyword found.

    Raises:
        KeyError: If an algorithm in keyword_dicts is not found in the weights dictionary.
    """
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

    for algorithm, keywords in keyword_dicts.items():
        # avoid the manually chosen topics being used in the common topics
        if algorithm == "chosen_topics":
            continue
        algorithm_weight = weights[algorithm]
        # ensure to only search for keywords that are not already in the frequency dictionary
        new_keywords = [
            key[0] for key in keywords if key[0] and key[0] not in frequency_dict
        ]
        frequency_dict.update(word_frequency(new_keywords, word_freq_dict))

        for rank, key_tuple in enumerate(keywords):
            keyword, keyword_score = key_tuple

            # Skip empty keywords
            if not keyword:
                continue
            # Calculate the weight based on the rank of the keyword
            highest = parameters["highest_rank"]
            rankweight = (highest - rank if rank < highest else 1) * parameters[
                "rank_weight"
            ]

            algoscale = {
                "spacy_NER": 1,  # no score given
                "rake_keywords": 10,
                "tf_IDF": 1,
                "LDA": 0.05,
                "NMF": 1,
            }

            score_weight = (
                keyword_score / algoscale[algorithm] * algorithm_weight
                + algorithm_weight
            )

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
                ) * parameters["position_weight"]

                # Assign a score based on the calculated weights
                keyword_score = (
                    rankweight
                    + score_weight
                    + position_weight
                    + frequency_weight
                    + digit_weight
                )
                # print algorithm and keyword, all weights and score
                # print(f"{algorithm}, {keyword}, {rankweight} + {score_weight} + {position_weight} + {frequency_weight} + {digit_weight} = {keyword_score}")

            else:
                # calculate the score without the position weight
                keyword_score = (
                    rankweight + algorithm_weight + frequency_weight + digit_weight
                )

            term_count[keyword] += keyword_score

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
    # TODO: Consider if it makes sense using the longest terms
    most_common_terms = []
    for term, keyword_score in sorted_terms:
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

    return most_common_terms[
        :10
    ]  # TODO: consider adding a parameter for the number of common topics


def remove_stopwords(text, nlp_spacy):
    # Tokenize the text using spaCy
    doc = nlp_spacy(text)

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


def filter_keywords(keywords, nlp_spacy, parameters):
    """
    Filters the given list of keywords based on specified conditions.

    Args:
        keywords (list): A list of keywords to be filtered.
        nlp_spacy (spacy.lang): A spaCy language model for processing keywords.

    Returns:
        list: A list of filtered keywords.
    """

    filtered_keywords = []
    lowercase_keywords = set()

    # only keywords with min length of 3
    keywords = [keyword for keyword in keywords if len(keyword) > 2]
    # Create spaCy tokens from the keywords
    batch_size = parameters["batch_size"]
    keywords_and_tokens = [
        (doc[0], keyword)
        for keyword, doc in zip(
            keywords, nlp_spacy.pipe(keywords, batch_size=batch_size)
        )
    ]

    filtered_keywords = [
        keyword
        for token, keyword in keywords_and_tokens
        if is_valid_keyword(keyword, token, lowercase_keywords)
    ]

    return filtered_keywords


def is_valid_keyword(keyword, token, lowercase_keywords):
    """
    Checks if the given keyword is valid based on specified conditions.

    Args:
        keyword (str): The keyword to be checked for validity.
        token (spacy.tokens.Token): The spaCy token corresponding to the keyword.

    Returns:
        bool: True if the keyword is valid, False otherwise.
    """
    # Find all sets of digits in the keyword
    digit_sets = re.findall(r"\d+", keyword)
    # Check if the keyword has only one set of digits with a maximum of 3 digits
    valid_digit_rule = len(digit_sets) <= 1 and all(len(ds) <= 3 for ds in digit_sets)
    # Check if the keyword contains a link
    contains_link = "http" in keyword.lower() or "www" in keyword.lower()

    # Check if the keyword is not already in the set
    not_duplicate = keyword.lower() not in lowercase_keywords

    return (
        len(keyword) > 2
        and not token.is_stop
        and valid_digit_rule
        and not contains_link
        and not_duplicate
    )


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


def preprocess_text(cleaned_texts, docs):
    """
    Preprocess the input texts using spaCy's NLP pipeline, removing stopwords and applying lemmatization.

    Parameters
    ----------
    text_list : list of str
        A list of strings representing the input texts to preprocess.
    nlp_spacy : spacy.lang model
        A preloaded spaCy language model (with unnecessary components disabled).

    Returns
    -------
    list of str
        A list of preprocessed texts, where stopwords have been removed and lemmatization has been applied.
    """

    preprocessed_texts = []

    # Apply SpaCy NLP pipeline to the texts in a batch
    # docs = nlp_spacy.pipe(text_list)

    for doc in docs:
        # Remove stopwords and apply lemmatization
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop]

        # Join the lemmatized tokens back into a single string
        preprocessed_text = " ".join(lemmatized_tokens)

        preprocessed_texts.append(preprocessed_text)

    return preprocessed_texts


def extract_keywords(cleaned_texts, parameters, nlp_spacy, stopwords):
    """
    Extracts keywords using various methods from a list of cleaned texts.

    Args:
        cleaned_texts (list): A list of cleaned texts.
        parameters (dict): A dictionary containing the parameters for each keyword extraction method.
        nlp_spacy: The spaCy language model for preprocessing.

    Returns:
        tuple: A tuple containing the extracted keywords from different methods:
            - spacy_keywords (list): Keywords extracted using spaCy NER.
            - rake_keywords (list): Keywords extracted using RAKE.
            - tf_IDF_keywords (list): Keywords extracted using TF-IDF.
            - LDA_keywords (list): Keywords extracted using LDA and sorted by input order.
            - NMF_keywords (list): Keywords extracted using NMF and sorted by input order.

    """

    if printss:
        print("rake1")
    rake_keywords = rake(cleaned_texts, parameters["rake"])
    if printss:
        print("rake done2")
    batch_size = parameters["spacy"]["batch_size"]
    spacy_docs = nlp_spacy.pipe(cleaned_texts, batch_size=batch_size)
    if printss:
        print("spacy")
    spacy_docs = list(nlp_spacy.pipe(cleaned_texts))
    spacy_keywords = spacy_ner(cleaned_texts, parameters["spacy"], spacy_docs)

    # remove stopwords and perform lemmatization
    # spacy_docs = nlp_spacy.pipe(cleaned_texts)
    cleaned_texts = preprocess_text(cleaned_texts, spacy_docs)
    if printss:
        print("tfidf")
    tf_IDF_keywords = tf_IDF(cleaned_texts, parameters["tf_IDF"])
    if printss:
        print("NMF")
    NMF_keywords = NMF_topic_modeling(cleaned_texts, parameters["NMF"])
    NMF_keywords = sort_keywords(NMF_keywords, cleaned_texts)  # TODO: move into function
    if printss:
        print("LDA")
    LDA_keywords = LDA_topic_modeling(cleaned_texts, parameters["LDA"])
    LDA_keywords = sort_keywords(LDA_keywords, cleaned_texts)  # TODO: move into function
    if printss:
        print("done")
    return spacy_keywords, rake_keywords, tf_IDF_keywords, LDA_keywords, NMF_keywords


def store_keywords(
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


def filter_keywords_tuples(keywords_tuples, nlp_spacy, parameters):
    # If keywords_tuples is an empty list, return an empty list
    if not keywords_tuples:
        return []
    
    # Separate keywords and values
    keywords, values = zip(*keywords_tuples)
    # Filter keywords
    filtered_keywords = filter_keywords(keywords, nlp_spacy, parameters)
    # Pair filtered keywords with their original values
    filtered_keywords_tuples = [(kw, v) for kw, v in zip(keywords, values) if kw in filtered_keywords]
    return filtered_keywords_tuples



def extract_common_topics(filtered_messages, parameters, word_freq_dict, nlp_spacy):
    """Extract common topics, using the function 'find_common_topics',
    from the messages and store them in the message dictionaries."""
    for message in filtered_messages:
        if "common_topics" in message["topic_suggestions"]:
            print("common topics already extracted: ", message["id"])
            continue

        # print("message: ", filter_string(message["message"][:first_letters]))

        common_topics = find_common_topics(
            message["topic_suggestions"],
            filter_string(
                message["message"][: parameters["extract_topic"]["first_letters"]]
            ),
            parameters["keyword_selection_parameters"],
            word_freq_dict,
        )
        common_topics = filter_keywords(
            common_topics, nlp_spacy, parameters["spacy_keywords"]
        )
        # add common topics to message
        message["topic_suggestions"]["common_topics"] = common_topics
        print("common topics: ", message["topic_suggestions"]["common_topics"])
        print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])
        message["topic_suggestions"]["spacy_NER"] = filter_keywords_tuples(message["topic_suggestions"]["spacy_NER"], nlp_spacy, parameters["spacy_keywords"])
        print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])
        # message["topic_suggestions"]["spacy_NER"] = filter_keywords(message["topic_suggestions"]["spacy_NER"], nlp_spacy, parameters["spacy_keywords"])
        # message["topic_suggestions"]["rake_keywords"] = filter_keywords(message["topic_suggestions"]["rake_keywords"], nlp_spacy, parameters["spacy_keywords"])
        # message["topic_suggestions"]["tf_IDF"] = filter_keywords(message["topic_suggestions"]["tf_IDF"], nlp_spacy, parameters["spacy_keywords"])
        # message["topic_suggestions"]["LDA"] = filter_keywords(message["topic_suggestions"]["LDA"], nlp_spacy, parameters["spacy_keywords"])
        # message["topic_suggestions"]["NMF"] = filter_keywords(message["topic_suggestions"]["NMF"], nlp_spacy, parameters["spacy_keywords"])
        
        # filter keywords from algorithms as well
        # for key, keywords in message["topic_suggestions"].items():
        #     if key == "chosen_topics":
        #         continue
        #     message["topic_suggestions"][key] = filter_keywords(
        #         keywords, nlp_spacy, parameters["spacy_keywords"]
        #     )






def extract_topic(filtered_messages, parameters, word_freq_dict):
    # add topic_suggestions key to each message
    for message in filtered_messages:
        message.setdefault("topic_suggestions", {})

    print("parameters: ", parameters)

    # check for predefined topic "Thema: ..."
    check_if_topic(filtered_messages)

    # Clean and preprocess the texts
    cleaned_texts_with_indices = [
        (
            idx,
            filter_string(
                message["message"][: parameters["extract_topic"]["first_letters"]]
            ),
        )
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
    ) = extract_keywords(
        [text for _, text in cleaned_texts_with_indices],
        parameters,
        nlp_spacy,
        stopwords,
    )
    store_keywords(
        filtered_messages,
        spacy_keywords,
        rake_keywords,
        tf_IDF_keywords,
        LDA_keywords,
        NMF_keywords,
        cleaned_texts_with_indices,
        nlp_spacy,
    )
    extract_common_topics(filtered_messages, parameters, word_freq_dict, nlp_spacy)


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
        # print("spacy_NER: ", message["topic_suggestions"]["spacy_NER"])

        # compare the common topics in 'topics' with the common topics in 'message'
        # step through each list in the dictionary
        for key, topics in message["topic_suggestions"].items():
            # skip the manually chosen topics
            if key == "chosen_topics":
                continue
            # step through each topic in the list
            for i, topic in enumerate(topics):
                if isinstance(topic, tuple):
                    topic = topic[0]
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
