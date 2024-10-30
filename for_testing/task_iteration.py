import sys
from typing import Dict, List
from prompts import Prompt
import pandas as pd
import requests

class Task_Iterations(Prompt):
    def __init__(self, prompt_examples: str) -> None:
        super().__init__(
            question_prefix = "",
            answer_prefix = "",
            intra_example_sep = "\n\n",
            inter_example_sep = "\n\n###\n\n",
        )
        self.count = 0
        self.prompt = self.make_prompt(prompt_examples = prompt_examples)

    def make_prompt(self, prompt_examples: str) -> str:
        prompt_examples = pd.read_json(prompt_examples, orient = "records")
        prompt_examples = prompt_examples[0:]
        grouped = prompt_examples.groupby("example")
        prompt = []
        # сортировка каждой группы по итоговой оценке
        for _, group in grouped:
            group["numerical_score"] = group["total_score"].apply(lambda x: int(x.split("/")[0].strip()))
            group = group.sort_values("numerical_score")
            prompt.append(self.make_one_iterate_example(group.to_dict("records")))
        return self.inter_example_sep.join(prompt) + self.inter_example_sep
        
    def make_one_iterate_example(self, incrementally_improving_examples: List[Dict]):
        """Тебе дан список примеров. Твоя задача - на основе спика вернуть новый пример."""
        
        instruction = """Твоя задача - последовательно улучшать качество представляемых ответов. Для каждого ответа приводятся оценки по желаемым характеристикам: 
        1) Релевантность, 
        2) Информативность, 
        3) Интересность содержания, 
        4) Последовательность, 
        5) Полезность, 
        6) Привлекательность, 
        7) Конкретность, 
        8) Безопасность, 
        9) Понимание пользователя, 
        10) Плавность.
        Вот несколько примеров таких оценок:"""
        
        template = """История диалога: {history}. Ответ: {response}.
        Полученные баллы:
        * Релевантность: {Relevant}     
        * Информативность: {Informative}
        * Интересность содержания: {Interesting}
        * Последовательность: {Consistent}
        * Полезность: {Helpful}
        * Привлекательность: {Engaging}
        * Конкретность: {Specific}
        * Безопасность: {Safe}
        * Понимание пользователя: {Userunderstanding}
        * Плавность: {Fluent}
        * Общая оценка: {total_score}
        Тебе нужно использовать эти баллы обратной связи для дальнейшего улучшения ответа."""
             
        prompt = []
        for row in incrementally_improving_examples:
            prompt.append(
                template.format(
                    history = row['history'].replace('System: ', '').replace('User: ', ''),
                    response = row["response"],
                    Relevant = row["Relevant"],
                    Informative = row["Informative"],
                    Interesting = row["Interesting"],
                    Consistent = row["Consistent"],
                    Helpful = row["Helpful"],
                    Engaging = row["Engaging"],
                    Specific = row["Specific"],
                    Safe = row["Safe"],
                    Userunderstanding = row["Userunderstanding"],
                    Fluent = row["Fluent"],
                    total_score = row["total_score"],))

        prompt = "".join(prompt)
        prompt = instruction + prompt
        return prompt.strip()

    def make_query(self, question: str) -> str:
        self.prompt = self.make_prompt(prompt_examples = "myfeedback.jsonl", reduce_window = 0)
        question = question.replace('System: ', '').replace('User: ', '')
        return f"{self.prompt}{self.question_prefix}{question}{self.intra_example_sep}{self.answer_prefix}"

    def _make_input(
        self,
        context: str,
        response: str,
        scores: str,
    ) -> str:
        context = context.replace('System: ', '').replace('User: ', '')
        input_txt = f"""История диалога: {context}. Полученный ответ: {response}. Баллы: {scores}. 
        Тебе нужно использовать эти баллы обратной связи для дальнейшего улучшения ответа."""
        return input_txt

    def __call__(
        self,
        responses_to_scores: Dict[str, str],
        reduce_window = 0
    ) -> str:
        example_input = self.make_input(
            responses_to_scores = responses_to_scores
        )
        transfer_query = self.make_query(example_input, reduce_window = reduce_window)
        self.count += 1
        with open(f"responses_iterate_{self.count}.txt", "w") as f:
            f.write(transfer_query + "\n")
        url = "http://sm-a5000-1.gpu.cluster:11435/api/chat"
        payload = {
                "model": "llama3.1:8b",
                "messages": transfer_query,
                "num_predict": 2000,
                "stop_token": self.inter_example_sep,
                "temperature": 0.1,
                "seed": 0,
                "stream": False,
                "mirostat": 0,
                "mirostat_eta": 0.1,
                "mirostat_tau": 5.0,
                "num_ctx": 2048,
                "repeat_last_n": 64,
                "repeat_penalty": 1.1,
                "tfs_z": 1,
                "top_k": 40,
                "top_p": 0.9,
                "min_p": 0.0,
                "num_keep": 4,
                "typical_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "penalize_newline": True,
                "numa": False,
                "num_batch": 512,
                "num_gpu": -1, 
                "main_gpu": 0,
                "low_vram": False,
                "f16_kv": True,
                "vocab_only": False,
                "use_mmap": False,
                "use_mlock": False,
                "num_thread": 0
        }  
        response = requests.post(url, json = payload)
        preliminary_feedback = response.json()["message"]["content"]
        response = preliminary_feedback.split("Response:")[1].strip().split("\n")[0].strip()
        return response.strip()

    def make_input(
        self,
        responses_to_scores: Dict[str, str],
    ) -> str:
        input_txt = ""
        for response, (context, scores) in responses_to_scores.items():
            context = context.replace('System: ', '').replace('User: ', '')
            input_txt += self._make_input(
                context = context,
                response = response,
                scores = scores,
            )
        return input_txt
    
if __name__ == "__main__":
    obj = Task_Iterations(prompt_examples = "myfeedback.jsonl")
    print(obj.prompt)