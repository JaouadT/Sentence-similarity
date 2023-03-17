"""Streamlit app to compute similarity between sentences or paragraphs."""

# Import from standard library
import logging
import random
import re

# Import from 3rd party libraries
import streamlit as st
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer, util

# Configure Streamlit page and state
st.set_page_config(page_title="Sentence Similarity", page_icon="🤖")

def compute_similarity(sentence1: str, sentence2: str, model) -> str:
    query_sentence = sentence1
    passage_sentences = sentence2
    query_embeddings = model.encode(query_sentence)
    passage_embedding = model.encode(passage_sentences)

    scores = util.cos_sim(query_embeddings, passage_embedding)
    return str(scores.numpy()[0][0])

st.write(
    """<style>
    [data-testid="column"] {
        width: calc(50% - 1rem);
        flex: 1 1 calc(50% - 1rem);
        min-width: calc(50% - 1rem);
    }
    </style>""",
    unsafe_allow_html=True,
)

# Render Streamlit page
st.title("Compute Similarity between sentences or paragraphs.")
st.markdown(
    "This mini-app computes similarity between two or multiple sentences using [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model."
)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentence1 = st.text_input(label="Sentence 1", placeholder="Text goes here...")
sentence2 = st.text_input(label="Sentence 2", placeholder="Text goes here...")

if st.button(
    label="Compute similarity",
    type="primary",
):
    result = compute_similarity(sentence1, sentence2, model)
    st.text(f'Sentence similarity: {result}')


text_spinner_placeholder = st.empty()


