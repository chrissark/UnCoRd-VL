import argparse
from UnCoRdVL import UnCoRdv2
from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UnCoRd test.')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--questions_file', type=str, help='A json file with questions, images indices and answers')
    parser.add_argument('--num_questions', type=int, default=None, help='Number of questions.')
    parser.add_argument('--test_mode', type=bool, default=False, help='Answers are not required in test mode.')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffles questions if true.')
    parser.add_argument('--device', type=str, default='cpu', help='Device for VL-BERT estimator.')
    args = parser.parse_args()
    questions, _ = get_questions_and_answers(args.questions_file, args.num_questions, test=args.test_mode, shuffle=args.shuffle)
    model = UnCoRdv2(args.device)
    with open('test_answers.txt', 'w') as f:
        for i, question in enumerate(questions):
            answer = model.get_answer(args.image_dir, question)
            print(f"Question: {question['question']}")
            print(f"Model answer: {answer}")
            print(f"Image id: {question['image_index']}")
            print(f"Question id: {question['question_index']}")
            print('\n')
            f.write(answer + '\n')