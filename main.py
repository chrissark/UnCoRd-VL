import argparse
from UnCoRdVL import UnCoRdv2
from dataset import *
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnCoRd test.')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--questions_file', type=str, help='A json file with questions, images indices and answers')
    parser.add_argument('--answer_vocab_file', type=str, help='Vocabulary for VL-BERT estimator.')
    parser.add_argument('--properties_file', type=str, help='Property categories of your dataset.')
    parser.add_argument('--num_questions', type=int, default=None, help='Number of questions.')
    parser.add_argument('--test_mode', type=bool, default=False, help='Answers are not required in test mode.')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffles questions if true.')
    parser.add_argument('--device', type=str, default='cpu', help='Device for VL-BERT estimator.')
    parser.add_argument('--question_index', type=int, default=None, help='If true, answers a single question with queried index.')
    args = parser.parse_args()
    if args.question_index:
        questions = []
        ques, _ = get_question_by_idx(args.questions_file, args.question_index, test=args.test_mode)
        questions.append(ques)
    else:
        questions, _ = get_questions_and_answers(args.questions_file, args.num_questions, test=args.test_mode, shuffle=args.shuffle)
    model = UnCoRdv2(args.device, args.answer_vocab_file, args.properties_file)
    with open('test_answers.txt', 'w') as f:
        for i, question in enumerate(questions):
            start = time.time()
            answer = model.get_answer(args.image_dir, question)
            end = time.time()
            print(f"Question: {question['question']}")
            print(f"Model answer: {answer}")
            print(f"Image id: {question['image_index']}")
            print(f"Question id: {question['question_index']}")
            print(f"VL-BERT calls: {model.num_vlbert_calls}")
            print(f"Wall time: {(end - start):.4f}s")
            print('\n')
            f.write(answer + '\n')