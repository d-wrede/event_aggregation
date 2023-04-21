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

# Load spaCy's German model
nlp = spacy.load("de_core_news_lg")

from rake_nltk import Rake

r = Rake(language="german")


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


def rake_keywords(message, max_length=3):
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


# def find_common_topics(keyword_dicts):
#     term_count = defaultdict(float)
#     longest_terms = {}
#     for algorithm, keywords in keyword_dicts.items():
#         for rank, keyword in enumerate(keywords):
#             # Assign a score based on the order of the keyword (higher rank = lower score)
#             score = 7 - rank if rank < 7 else 1

#             term_count[keyword] += score

#             # If the keyword is a substring of a longer term, update the longest_terms dictionary
#             if keyword in longest_terms:
#                 if (
#                     len(longest_terms[keyword]) < len(keyword)
#                     and len(keyword.split()) <= 2
#                 ):
#                     longest_terms[keyword] = keyword
#             else:
#                 if len(keyword.split()) <= 2:
#                     longest_terms[keyword] = keyword

#     # Sort the terms by their score in descending order
#     sorted_terms = sorted(term_count.items(), key=lambda x: x[1], reverse=True)

#     # Create a list of the most common terms, using the longest form of the term
#     most_common_terms = []
#     for term, score in sorted_terms:
#         if term in longest_terms:
#             most_common_terms.append(longest_terms[term])

#     return most_common_terms

# def find_common_topics(keyword_dicts):
#     term_count = defaultdict(float)
#     longest_terms = {}
#     for algorithm, keywords in keyword_dicts.items():
#         for rank, keyword in enumerate(keywords):
#             # Assign a score based on the order of the keyword (higher rank = lower score)
#             score = 7 - rank if rank < 7 else 1

#             term_count[keyword] += score

#             # If the keyword is a substring of a longer term, update the longest_terms dictionary
#             if keyword in longest_terms:
#                 if (
#                     len(longest_terms[keyword]) < len(keyword)
#                     and len(keyword.split()) <= 2
#                 ):
#                     longest_terms[keyword] = keyword
#             else:
#                 if len(keyword.split()) <= 2:
#                     longest_terms[keyword] = keyword

#     # Sort the terms by their score in descending order
#     sorted_terms = sorted(term_count.items(), key=lambda x: x[1], reverse=True)

#     # Create a list of the most common terms, using the longest form of the term
#     most_common_terms = []
#     for term, score in sorted_terms:
#         if term in longest_terms:
#             is_subset = False
#             for i, common_term in enumerate(most_common_terms):
#                 if term in common_term:
#                     is_subset = True
#                     break
#                 elif common_term in term:
#                     most_common_terms[i] = term
#                     is_subset = True
#                     break

#             if not is_subset:
#                 most_common_terms.append(longest_terms[term])

#     return most_common_terms


def find_common_topics(keyword_dicts, text):
    term_count = defaultdict(float)
    longest_terms = {}
    text_length = len(text)

    for algorithm, keywords in keyword_dicts.items():
        for rank, keyword in enumerate(keywords):
            # Find the index position of the keyword in the text
            index_position = text.find(keyword)

            if index_position != -1:
                # Calculate the position-based weight
                position_weight = 1 - (index_position / text_length)*2

                # Assign a score based on the order of the keyword (higher rank = lower score) and position weight
                score = (7 - rank if rank < 7 else 1) + position_weight*70
                print(f"score: {score} for {keyword}")
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
            else:
                if len(keyword.split()) <= 2:
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


def remove_stopwords(text):
    # Tokenize the text using spaCy
    doc = nlp(text)

    # Remove stop words and create a list of filtered tokens
    filtered_tokens = [token.text for token in doc if not token.is_stop]

    # Join the filtered tokens to create the cleaned text
    cleaned_text = " ".join(filtered_tokens)

    return cleaned_text


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
