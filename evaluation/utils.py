import re
from io import StringIO
import tokenize
import tiktoken
import transformers
import ollama
import hashlib
import asyncio
from openai import OpenAI,AsyncOpenAI
from configs import *
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

# chat_tokenizer_dir = "./deepseek_v2_tokenizer"
#
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#         chat_tokenizer_dir, trust_remote_code=True
#         )
# async query

# async version of query
import random
import asyncio
from openai import AsyncOpenAI

async_clients_cache = {}  # avoid re-creating clients

async def async_query(prompt: str, history_msg=[], model="deepseek-flash", retries=3, base_delay=1):
    """
    Asynchronous OpenAI-compatible API request with retries.
    """
    if model not in models_config:
        raise Exception("model not supported")
    model_info = models_config[model]

    # cache AsyncOpenAI clients
    if model not in async_clients_cache:
        async_clients_cache[model] = AsyncOpenAI(api_key=model_info["key"], base_url=model_info["base_url"])
    client = async_clients_cache[model]

    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=history_msg + [{"role": "user", "content": prompt}],
                temperature=0.0,
                # extra_body={"thinking": {"type": "disabled"}}
            )
            history_msg.append({"role": "user", "content": prompt})
            history_msg.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content, history_msg
        except Exception as e:
            wait_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"[async_query] Error: {e}, retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
    return "", None


def query(prompt:str, history_msg=[], model="deepseek-flash", uplimit=65000):
    try:
        # if len(prompt) > int(uplimit/0.4):
        #     encoded_input = tokenizer(prompt, return_tensors="pt")
        #     input_ids = encoded_input["input_ids"][0]
        #     max_length = uplimit
        #     truncated_input_ids = input_ids[:max_length]
        #     prompt = tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
        
        if model in models_config:
            model_info = models_config[model]
        else:
            raise Exception("model not supported")
        
        client = OpenAI(api_key=model_info["key"], base_url=model_info["base_url"])
        
        if model in {"c7b", "c13b", "codequen-7b"}:
            response = ollama.generate(model=model, prompt=prompt, options={"num_ctx": 8192})
            return response.response, history_msg
        if "max_token" in model_info:
            response = client.chat.completions.create(
                model=model,
                messages= history_msg +
                [{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
                max_tokens=model_info["max_token"],
                # extra_body={"thinking": {"type": "disabled"}}
            )
        else:
            response = client.chat.completions.create(
                model=model,
                # messages= [{"role": "system", "content": system_prompt_deepseek}] + history_msg +
                # [{"role": "user", "content": prompt}],
                messages= history_msg +
                [{"role": "user", "content": prompt}],
                stream=False,
                temperature=0.0,
                # xtra_body={"thinking": {"type": "disabled"}}
            )
        
        # print(f"Current token usage: {response.usage}")
        history_msg.append({"role": "user", "content": prompt})
        history_msg.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content, history_msg
    
    except Exception as e:
        print(e)
        return "", None

def prompt_summarize(func, language):
    tmp_prefix = f"""Summarize the provided function without external explanation:\n"""
    tmp_posfix = ""
    return tmp_prefix + func + tmp_posfix

def prompt_generate_code_fewshots(description, examples, language):
    tmp_prefix = """Generate a C language function based on the provided description and examples. Only create the requested function itself, without any additional functions, definitions, or main function.\n""" 
    descrition_part = f"# Description:\n{description}\n"
    shots = " ".join([f"## Example {i}:\n {x}\n" for i, x in enumerate(examples)])
    few_shots_part = f"# Examples:\n{shots}"
    tmp_posfix = f"\nReturn the function surrounded with ```{language}\\n```"
    return tmp_prefix + descrition_part + few_shots_part + tmp_posfix

def prompt_root_cause(CWE, description, full_diff, language):
    CWE_str = ", ".join(CWE)
    prefix = f"I will provide a fixing commit for a {language} language {CWE_str} vulnerability, along with its description and commit message. Please help me identify the core vulnerable logic patterns that introduce this vulnerability. These patterns should be general enough to help identify similar vulnerabilities in other code. Please output the patterns with example code snippet from orignal code one by one."
    descrition_part = f"\n# Description:\n```{description}\n```"
    diff_part = f"\n# Diff:\n```{full_diff}\n```"
    pos_fix = """
    The sample output format is:

# Pattern name
 - Pattern:
 - Example:
...

Don't output extra information.
    """
    return prefix + descrition_part + diff_part + pos_fix

def prompt_root_cause_2(CWE, description, full_diff, language):
    CWE_str = ", ".join(CWE)
    prefix = f"I will provide a fixing commit for a {language} language {CWE_str} vulnerability, along with its vulnerability description and commit details. Please help me identify the core vulnerable logic patterns and their corresponding fixes from the commit. These patterns should be general enough to help identify similar vulnerabilities in other code. Please analyze each pattern and provide example code snippets from the original code."
    descrition_part = f"\n# Description:\n```{description}\n```"
    diff_part = f"\n#  Fixing Commit:\n```{full_diff}\n```"
    pos_fix = """
    The sample output format is:

# Pattern name
 - Vulnerable Pattern:
 - Vulnerable Example:
 - Fixing Pattern:
 - Fixing Example:
...

Don't output extra information.
    """
    return prefix + descrition_part + diff_part + pos_fix

def prompt_root_cause_3(CWE, description, full_diff, language):
    CWE_str = ", ".join(CWE)
    prefix = f"I will provide a fixing commit for a {language} language {CWE_str} vulnerability, along with its vulnerability description. Please help me identify the core vulnerable logic patterns. These patterns should be general but include some details to help identify similar vulnerabilities in other code."
    descrition_part = f"\n# Description:\n```{description}\n```"
    diff_part = f"\n#  Fixing Commit:\n```{full_diff}\n```"
    pos_fix = """
    The sample output format is:

# Pattern name
 - Vulnerable Pattern:
...

Don't output extra information.
    """
    return prefix + descrition_part + diff_part + pos_fix


def prompt_judge_security_patterns(patterns, code):
    prefix = "You are an expert in code review and computer security. You will analyze two items: (1) Patterns: describing vulnerable actions, and (2) Code: the code to be reviewed. Your task is to determine whether the vulnerable patterns appear in the given code."
    posfix = """
    For each vulnerability pattern, indicate whether it exists in the code by a json format: 
{pattern name: "found" or "not found"}. Please only output whether each pattern exists in the code.
    """    
    return prefix + "\nPatterns:\n" + patterns + "\nCode:\n" + code + posfix

def prompt_judge_security_patterns_2(patterns, code):
    prefix = "You are an expert in code review and computer security. You will analyze two items: (1) Patterns: describing vulnerable actions with corresponding fixing actions. (2) Code: the code to be reviewed. Your task is to determine whether the given code is vulnerable. If the given code contains vulnerable patterns and their fixing patterns exist in the code, the code is considered as vulnerable."
    posfix = """
    For each vulnerability pattern, indicate in JSON format whether it exists and remains unfixed in the code. If the pattern is exist and not fixed, please output {pattern name: "found"}, else {pattern name: "not found"}.
    Please only output whether each pattern unfixed in the code.
    """    
    return prefix + "\nPatterns:\n" + patterns + "\nCode:\n" + code + posfix

def prompt_judge_security_patterns_3(patterns, code):
    prefix = "You are an expert in code review and computer security. You will analyze two items: (1) Patterns: describing vulnerable actions. (2) Code: the code to be reviewed. Your task is to determine whether the given code is vulnerable. If the given code contains vulnerable patterns and is not fixed, the code is considered as vulnerable."
    posfix = """
    For each vulnerability pattern, indicate in JSON format whether it exists and remains unfixed in the code. If the pattern is exist and not fixed, please output {pattern name: "found"}, else {pattern name: "not found"}.
    Please only output whether each pattern unfixed in the code.
    """    
    return prefix + "\nPatterns:\n" + patterns + "\nCode:\n" + code + posfix

def prompt_judge_security_chain_rb(code):
    role = "Please describe the intent of the given code. \n"
    basic = """\n I want you to act as a vulnerability detection system. Is the above program buggy? Please output the result by a json format: {Vulnerability: "found" or "not found"}"""
    return role + "\nCode:\n" + "```\n" + code + "\n```\n" + basic

def extract_code_blocks(markdown):
    code_blocks = []
    code_pattern = r'```[\s\S]*?```'
    code_matches = re.findall(code_pattern, markdown)
    for code_match in code_matches:
        code = code_match.strip('`')
        code = re.sub(r'^\w+', '', code).strip()
        code_blocks.append(code)
    return code_blocks

def is_equal(code_a, code_b, language):
    try:
        code_a = remove_comments_and_docstrings(code_a, language)
        code_b = remove_comments_and_docstrings(code_b, language)
    except:
        return False
    return remove_spaces(code_a) == remove_spaces(code_b)

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp=[]
        for x in out.split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp=[]
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip()!="":
                temp.append(x)
        return '\n'.join(temp)

def cnt_length(text):
    return len(tokenizer.encode(text))

def num_tokens_from_string(string: str) -> int:
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def remove_spaces(text):
    return re.sub(r'\s+', '', text)

def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()


def evaluate_binary_classification(y_true, y_pred):
    # compute the metrics
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # return the results
    metrics = {
        "accuracy": accuracy,
        "precision": prec,
        "recall": recall,
        "F1": f1
    }
    return metrics

if __name__ == "__main__":
    tmp_str = "Hello, world!"
    print(cnt_length(tmp_str))