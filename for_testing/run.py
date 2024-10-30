import re
import sys
import math
import os
import tqdm
from typing import Any, Dict, List
import pandas as pd
import json
from tqdm import tqdm
from pandarallel import pandarallel
import multiprocessing
import traceback
import argparse
import random
import time
from task_initialization import Task_Initialization
from task_iteration import Task_Iterations
from generate_feedback import Generation_of_Feedback
from prompts import retry_parse_fail_prone_cmd

pandarallel.initialize(progress_bar = True)

@retry_parse_fail_prone_cmd
def iterative_response(context: str, max_attempts: int) -> str:
    # генерация первого ответа LLM
    task_init = Task_Initialization(prompt_examples = ".for_testing/prompts/myinit.jsonl")
    # получение обратной связи
    task_feedback = Generation_of_Feedback(prompt_examples = ".for_testing/prompts/myfeedback.jsonl")
    # итеративное улучшение ответа
    task_iterate = Task_Iterations(prompt_examples = ".for_testing/prompts/myfeedback.jsonl")
    
    # инициализация задачи
    n_attempts = 0
    responses_to_scores = dict()
    all_responses_to_scores = dict()
    best_score_so_far = 0
    reduce_window = 0
    while n_attempts < max_attempts:
        if n_attempts == 0:
            metaoutput, response = task_init(context = context)
        else:
            metaoutput, response = task_iterate(responses_to_scores = responses_to_scores, reduce_window = reduce_window)

        print(f"\n{n_attempts} КОНТЕКСТ> {context} \n\n ОТВЕТ> {response} - NTOKENS> {metaoutput['usage']['total_tokens']}")
        
        if metaoutput['usage']['total_tokens'] > 3000:
            reduce_window += 1
            if metaoutput['usage']['total_tokens'] > 3500:
                reduce_window += 1

        feedbackmetaoutput, scores = task_feedback(context = context, response = response)
        print(f"\n{n_attempts} БАЛЛЫ> {scores} - NTOKENS> {feedbackmetaoutput['usage']['total_tokens']}")

        total_score = re.search(r"Итог: (\d+)/(\d+)", scores).group(0)
        total_score = int(total_score.split(":")[1].strip().split("/")[0])
        
        all_responses_to_scores[response] = {
            "n_attempts": n_attempts,
            "scores": scores,
            "total_score": total_score,
            "context": context,
        }
        if total_score >= 0:  # имеет смысл итерироваться, если процесс идет на улучшение
            best_score_so_far = total_score
            responses_to_scores[response] = (context, scores)
        else:
            print(f"Оценка {response} составляет {total_score}, что меньше текущего лучшего значения, которое равно {best_score_so_far}.")
        n_attempts += 1
    return all_responses_to_scores

def run_dataset(max_attempts: int, outfile: str, max_size: int = 1):
    f = open(r"C:\Users\julia\OneDrive\Desktop\MFTI_code\prompting\for_testing\prompts\my_fed_data.json")
    data = json.load(f)
    print('Количество данных:', len(data))
    count = 0
    outwriter = open(outfile, 'a')
    for i, example in enumerate(data[:]):
        if max_size != 0 and count > max_size: break
        print(f"\n\n\n****Итерация: {i}****\n\n")
        if "response" not in example: 
            continue
        try:
            context = example["context"]
            if type(example["context"]) is str:
                context = example["context"].split("\n")
            if type(context) is list:
                context = "\n".join(context[-8:])
            all_responses_to_scores = iterative_response(context, max_attempts = max_attempts)
            if all_responses_to_scores is None:
                return {"Результат": ["ПРОВАЛЕНО"]}
            
            res = []
            scored_responses = {}
            for response, scores in all_responses_to_scores.items():
                res.append(f"{response} [общее количество баллов: {scores['total_score']}] \n {scores['scores']}")
                scored_responses[scores['n_attempts']] = {'response':response, 'total_score':scores['total_score']}
            example['generated_responses'] = "\n------\n".join(res)
            example['scored_responses'] = scored_responses
            outwriter.write(json.dumps(example)+'\n')
            print("\n ------ \n ".join(res))
        except Exception as e:
            print(f"ошибка в {example}\n\n{e}", file = sys.stderr)
            traceback.print_exc()
            return {"result": ["FAILED"]}
        count += 1
    outwriter.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_attempts",
        type = int,
        default = 3,
        help = "Max attempts",
    )
    parser.add_argument(
        "--size",
        type = int,
        default = 1,
        help = "Test data size (0 means all data)",
    )
    parser.add_argument(
        "--output",
        type = str,
        default ='./output-v3fedresponsegen406on.json',
        help = "Output file",
    )
    args = parser.parse_args()
    run_dataset(args.max_attempts, outfile = args.output, max_size = args.size)