import random
import os
import json


def get_questions_and_answers(questions_file, num_questions=None, test=False, shuffle=False, start_index=0):
    with open(questions_file, 'r') as f:
        q_info = json.load(f)

    q_info = q_info['questions']
    answers = []
    questions = []
    for i, ques in enumerate(q_info):
        questions.append(ques)

    if not num_questions:
        num_questions = len(questions)

    if shuffle:
        sample = random.sample(questions, num_questions)
    else:
        sample = questions[start_index:num_questions]
    if not test:
        for ques in questions:
            answers.append(ques['answer'])

    return sample, answers


def get_question_by_idx(questions_file, ques_id):
    with open(questions_file, 'r') as f:
        q_info = json.load(f)

    q_info = q_info['questions']

    question = q_info[ques_id]
    answer = question['answer']

    return question, answer
