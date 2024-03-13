# -*- coding: utf-8 -*-
from typing import List

import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from spacy import Language
from spacy_langdetect import LanguageDetector

MAX_CHARACTERS = 1000000


# The unused parameters required by spacy_langdetect
def get_lang_detector(nlp, name):
    return LanguageDetector()


class KeywordExtractor:
    def __init__(self, model_name: str, nb_keywords=10):
        # used for EN lemmatization and language detection,
        # this is said to be thread-safe - see https://stackoverflow.com/a/63447151/1263942
        self.nlp_en_core_web_sm = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        self.nlp_en_core_web_sm.add_pipe("language_detector", last=True)
        self.nlp_en_core_web_sm.max_length = MAX_CHARACTERS

        # used for FR lemmatization
        self.nlp_fr_core_news_sm = spacy.load("fr_core_news_sm")
        self.nlp_fr_core_news_sm.max_length = MAX_CHARACTERS

        self.nb_keywords = nb_keywords
        self.kw_model = KeyBERT(model=SentenceTransformer(model_name))

    def __call__(self, text: str) -> List[str]:
        language = self.detect_language(text)
        keywords = self.extract_keywords(text)

        return self.lemmatize(keywords, language)

    def extract_keywords(self, text: str) -> List[str]:
        keywords_scores = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            top_n=self.nb_keywords,
            stop_words=None,
        )

        return [keyword for keyword, _ in keywords_scores]

    def detect_language(self, text: str):
        # noinspection PyProtectedMember
        return self.nlp_en_core_web_sm(text)._.language["language"]

    def lemmatize(self, keywords: List[str], language: str) -> List[str]:
        # concatenate all keywords in a sentence
        keywords_text = " ".join(keywords)

        if language == "fr":
            doc = self.nlp_fr_core_news_sm(keywords_text)
        else:
            doc = self.nlp_en_core_web_sm(keywords_text)
        # remove duplicates and short words
        return list({token.lemma_ for token in doc if len(token.lemma_) >= 3})
