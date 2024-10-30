import pandas as pd
from prompts import Prompt
from typing import List, Optional, Union
import sys
import requests

class Task_Initialization(Prompt):
    def __init__(self, prompt_examples: str, numexamples = 3) -> None:
        super().__init__(
            question_prefix ="История разговора: ",
            answer_prefix = "Ответ: ",
            intra_example_sep = "\n\n",
            inter_example_sep = "\n\n###\n\n",
        )
        self.setup_prompt_from_examples_file(prompt_examples, numexamples = numexamples)

    def setup_prompt_from_examples_file(self, examples_path: str, numexamples = 10) -> str:
        instruction = ("""Тебе дан диалог двух людей. Твоя задача - сгенерировать ответ, максимально согласующийся с историей диалога. Сгенерированный тобою ответ должен полностью соответствовать следующим критериям:
        1) Релевантность -  ответ соответствует предоставленному контексту диалога; 
        2) Информативность - ответ предоставляет информацию, необходимую для продолжения диалога;
        3) Интересность содержания - ответ интересен предполагаемому собеседнику или пользователю; 
        4) Последовательность - ответ соответствует стилю предоставленного диалога;
        5) Полезность - ответ предоставляет полезную собеседнику или пользователю информацию или содержит предложения каких-либо действий;
        6) Привлекательность - ответ вовлекает собеседника в диалог и стимулирует дальнейший разговор;
        7) Конкретность - содержание ответа предельно конкретно;
        9) Понимание пользователя - твой ответ должен показывать понимание темы и содержания диалога с пользователем;
        10) Плавность. 
        Твой ответ должнен начинаться ТОЛЬКО с символов:\n\n"
        """)

        examples_path = "C:\\Users\\julia\\OneDrive\\Desktop\\MFTI_code\\prompting\\for_testing"
        examples_df = pd.read_json(examples_path, orient = "records", lines = True)
        prompt = []
        for i, row in examples_df.iterrows():
            if i >= numexamples:
                break
            else:
                prompt.append(self._build_query_from_example(row["history"], row["response"]))

        self.prompt = instruction + self.inter_example_sep.join(prompt) + self.inter_example_sep

    def _build_query_from_example(self, history: Union[str, List[str]], response: Optional[str] = None) -> str:
        history = history.replace('System: ', '').replace('User: ', '')
        TEMPLATE = """История диалога: {history}. Ответ: {response}"""
        query = TEMPLATE.format(history = history, response = response)
        return query

    def make_query(self, context: str) -> str:
        context = context.replace('System: ', '').replace('User: ', '')
        query = f"{self.prompt}{self.question_prefix}\n\n{context}{self.intra_example_sep}"
        return query

    def __call__(self, context: str) -> str:
        generation_query = self.make_query(context)
        url = "http://sm-a5000-1.gpu.cluster:11435/api/chat"
        payload = {
                "model": "llama3.1:8b",
                "messages": generation_query,
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
        generated_response = preliminary_feedback.split(self.answer_prefix)[1].replace("#", "").strip()
        return generated_response.strip()