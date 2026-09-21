#!/usr/bin/env python3
"""
Module defining answer_loop function for interactive QA.
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in an interactive loop.

    Args:
        reference (str): The reference document containing answers.
    """
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        try:
            question = input("Q: ")
        except (KeyboardInterrupt, EOFError):
            print("A: Goodbye")
            break

        if question.strip().lower() in exit_commands:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)

        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
