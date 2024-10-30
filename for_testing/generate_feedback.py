from prompts import Prompt
import pandas as pd
import requests

class Generation_of_Feedback(Prompt):
    def __init__(self, engine: str, prompt_examples: str, max_tokens: int = 4000) -> None:
        super().__init__(
            question_prefix="",
            answer_prefix="",
            intra_example_sep="\n\n",
            inter_example_sep="\n\n###\n\n",
        )
        self.max_tokens = max_tokens
        self.setup_prompt_from_examples_file(prompt_examples)

    def setup_prompt_from_examples_file(self, examples_path: str) -> str:
        template = """История диалога: {history}. Полученный ответ: {response}. Полученные баллы:
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
        * Общая оценка: {total_score}"""
        examples_path = "C:\\Users\\julia\\OneDrive\\Desktop\\MFTI_code\\prompting\\for_testing"
        examples_df = pd.read_json(examples_path)
        prompt = []
        for _, row in examples_df.iterrows():
            prompt.append(template.format(
                    history = row['history'].replace('System: ', '').replace('User: ', ''),
                    response = row["response"],
                    Relevant = row["Релевантность"],
                    Informative = row["Информативность"],
                    Interesting = row["Интересность содержания"],
                    Consistent = row["Последовательность"],
                    Helpful = row["Полезность"],
                    Engaging = row["Привлекательность"],
                    Specific = row["Конкретность"],
                    Safe = row["Безопасность"],
                    Userunderstanding = row["Понимание пользователя"],
                    Fluent = row["Плавность"],
                    total_score = row["Общая оценка"])
            )
        instruction = """Твоя задача - последовательно улучшать качество представляемых ответов. Для каждого ответа приводятся оценки по желаемым характеристикам: 1) Релевантность, 2) Информативность, 
        3) Интересность содержания, 4) Последовательность, 5) Полезность, 6) Привлекательность, 7) Конкретность, 8) Безопасность, 9) Понимание пользователя и 10) Плавность.
        Вот несколько примеров таких оценок."""
        self.prompt = instruction + self.inter_example_sep.join(prompt)
        self.prompt = self.inter_example_sep.join(prompt) + self.inter_example_sep
    
    def __call__(self, context: str, response: str):
        prompt = self.get_prompt_with_question(context = context, response = response)
        url = "http://sm-a5000-1.gpu.cluster:11435/api/chat"
        payload = {
                "model": "llama3.1:8b",
                "messages": prompt,
                "num_predict": 2000,
                "stop_token": "###",
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
        generated_feedback = preliminary_feedback.split("Финальная оценка:")[1].strip()
        generated_feedback = generated_feedback.split("#")[0].strip()
        return generated_feedback

    def get_prompt_with_question(self, context: str, response: str):
        context = context.replace('System: ', '').replace('User: ', '')
        question = self.make_query(context = context, response = response)
        return f"""{self.prompt}{question}\n\n"""

    def make_query(self, context: str, response: str):
        question = f"""История разговора:{context}. Ответ: {response}"""
        return question