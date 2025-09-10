import json
import os
import re
import requests
import time
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
import asyncio
import aiohttp
import argparse

class DeepSeekAPI:
    def __init__(self, api_key, model="deepseek-chat", max_concurrency=10):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1" # if you use gpt, please change 
        )
        self.model = model
        self.max_concurrency = max_concurrency

    def generate(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = 30 * (attempt + 1) 
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"API Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    if attempt == max_retries - 1:
                        return ""
        return ""

    async def generate_one(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if "rate limit" in str(e).lower():
                        print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    else:
                        print(f"Error: {str(e)}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return ""
        return ""

    async def generate_batch(self, prompts):
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def limited_generate(prompt):
            async with semaphore:
                return await self.generate_one(prompt)

        tasks = [limited_generate(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

class DataProcessor:
    def __init__(self, input_file, retrieved_num, target=None):
        self.input_file = input_file
        self.retrieved_num = retrieved_num
        self.target = target

    def extract_code_snippet(self, ctx_text):
        code_match = re.search(r'<CODESPLIT>(?:[^<]*<CODESPLIT>){3}(.*)', ctx_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return ""

    def process(self):
        with open(self.input_file, 'r') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            question = {token.lower() for token in item["question"].split()}
            if self.target and self.target not in question:
                continue

            code_snippets = []
            for ctx in item["ctxs"][:self.retrieved_num]:
                code = self.extract_code_snippet(ctx["text"])
                code_snippets.append(code if code else "none")

            while len(code_snippets) < self.retrieved_num:
                code_snippets.append('none')

            processed_data.append({
                "question": item["question"],
                "retrieved_snippets": code_snippets
            })

        return processed_data

class CodeGenerator:
    def __init__(self, api_key, retrieved_num=10, model="deepseek-coder", max_concurrency=10):
        self.api = DeepSeekAPI(api_key, model, max_concurrency=max_concurrency)
        self.retrieved_num = retrieved_num


    def build_prompt(self, question, code_snippets):
        valid_snippets = [(i + 1, snippet) for i, snippet in enumerate(code_snippets[:self.retrieved_num]) if snippet != "none"]

        prompt_parts = []
        prompt_parts.append("Output only the code without any explanations, comments, or extra text.")

        for idx, snippet in valid_snippets:
            prompt_parts.append(f"<NL {idx}> \n {question.strip()} </NL {idx}>")
            prompt_parts.append(f"<Code Snippet {idx}> \n {snippet.strip()} </Code Snippet {idx}>")

        prompt_parts.append(f"<NL> \n {question.strip()} </NL>")
        prompt_parts.append("<Code Snippet>")

        return '\n'.join(prompt_parts)

    async def generate(self, data_batch):

        prompts = [self.build_prompt(item["question"], item["retrieved_snippets"]) for item in data_batch]
        generated_codes = await self.api.generate_batch(prompts)

        for i, generated_code in enumerate(generated_codes):
            data_batch[i]["generated_code"] = self.clean_generated_code(generated_code)
            data_batch[i]["prompt"] = prompts[i]  # 保存提示

        return data_batch

    def clean_generated_code(self, code):
        code = re.sub(r'```(python)?|```', '', code)
        if "def " in code:
            code = code[code.index("def "):]
        elif "class " in code:
            code = code[code.index("class "):]

        return code.strip()


class ResultSaver:
    @staticmethod
    def save(data, output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            for item in data:
                result = {
                    "question": item["question"],
                    "generated_code": item["generated_code"]
                }
                f.write(json.dumps(result) + '\n')

    @staticmethod
    def save_as_python_files(data, output_dir,retrieval_num=None):
        if retrieval_num is not None:
            output_dir = os.path.join(output_dir, f"retrieval_{retrieval_num}")

        os.makedirs(output_dir, exist_ok=True)


        for i, item in enumerate(data):
            file_name = f"generate_{i}.py"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(item["generated_code"])

        print(f"Saved {len(data)} Python files to {output_dir}")


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.processor = DataProcessor(
            input_file=config["input_file"],
            retrieved_num=config.get("retrieved_num", 5),
            target=config.get("target", None)
        )
        self.generator = CodeGenerator(
            api_key=config["api_key"],
            retrieved_num=config.get("retrieved_num", 5),
            model=config.get("model", "deepseek-coder"),
            max_concurrency = config.get("max_concurrency", 10)
        )
        self.batch_size = config.get("batch_size", 100) 
        self.retrieval_num = config.get("retrieved_num", 5)

    def run(self, output_file):

        print("Processing input data...")
        processed_data = self.processor.process()
        total_items = len(processed_data)
        print(f"Processed {total_items} items.")

        results = []
        print("Generating code via API...")

        with tqdm(total=total_items, desc="Generating Code") as pbar:
            for i in range(0, total_items, self.batch_size):
                batch = processed_data[i:i + self.batch_size]

                for item in batch:
                    item["prompt"] = self.generator.build_prompt(
                        item["question"],
                        item["retrieved_snippets"]
                    )

                generated_batch = self.generator.generate(batch, pbar)
                results.extend(generated_batch)

                if i % (self.batch_size * 10) == 0:
                    self.save_partial(results, output_file)

        self.save_partial(results, output_file, final=True)

        return results

    async def run_async(self, output_file):
        print("Processing input data...")
        processed_data = self.processor.process()
        total_items = len(processed_data)
        print(f"Processed {total_items} items.")

        results = []
        print("Generating code via DeepSeek API asynchronously...")

        pbar = tqdm(total=total_items, desc="Generating Code")

        for i in range(0, total_items, self.batch_size):
            batch = processed_data[i:i + self.batch_size]

            batch = await self.generator.generate(batch)
            results.extend(batch)

            pbar.update(len(batch))
            pbar.set_postfix_str(f"Processed {min(i + self.batch_size, total_items)}/{total_items}")

            if i % (self.batch_size * 10) == 0:
                self.save_partial(results, output_file)

        self.save_partial(results, output_file, final=True)
        pbar.close()


    def save_partial(self, results, output_file, final=False):
        temp_file = output_file.replace(".jsonl", "_temp.jsonl")
        ResultSaver.save(results, temp_file)

        if final:
            os.rename(temp_file, output_file)
            print(f"Final results saved to {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run RAG pattern code generation with DeepSeek API")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input JSON file containing retrieval results")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output JSONL file for generated results")
    parser.add_argument("--target", type=str, default=None,
                        help="Target keyword to filter questions (default: None)")
    parser.add_argument("--api_key", type=str,help="Api key for accessing the LLM service")

    args = parser.parse_args()


    config = {
        "input_file": args.input_file,
        "output_file": args.output_file,
        "api_key": args.api_key,  
        "model": "deepseek-chat",  # deepseek-coder, deepseek-coder-instruct, if you use gpt please change 
        "retrieved_num": 10,
        "batch_size": 200,  
        "max_concurrency": 40,  
        "target": args.target
    }


    pipeline = Pipeline(config)
    asyncio.run(pipeline.run_async(config["output_file"]))



