#!/usr/bin/env python3
"""
Multi-reference Question Answering loop using semantic search and BERT.
"""
question_answer_single = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from a corpus of documents in an interactive loop.

    Args:
        corpus_path (str): Path to the corpus of reference documents.
    """
    farewells = ["exit", "quit", "goodbye", "bye"]

    while True:
        try:
            user_input = input("Q: ")
        except (KeyboardInterrupt, EOFError):
            print("A: Goodbye")
            break

        if user_input.strip().lower() in farewells:
            print("A: Goodbye")
            break

        # 1. Search for the most relevant reference document
        reference = semantic_search(corpus_path, user_input)

        if reference is None:
            print("A: Sorry, I do not understand your question.")
            continue

        # 2. Extraction of the answer from reference document using BERT
        answer = question_answer_single(user_input, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
