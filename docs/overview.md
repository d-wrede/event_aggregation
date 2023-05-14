optimize_parameters:
    - cma call
    - process_messages(main.py) -> extract_topic(.py) -> 
    - evaluate_topic_extraction(.py)

extract_topic:
- check_if_topic(filtered_messages)
- filter_string(messages)
- extract_keywords
  - rake
  - spacy_ner
  - preprocess texts (using spacy): ```docs = nlp_spacy.pipe(text_list)```
  - preprocess_text (remove stop words and lemmatize words)
  - NMF_topic_modeling
  - LDA_topic_modeling
- store_keywords
- extract_common_topics
  - find_common_topics
