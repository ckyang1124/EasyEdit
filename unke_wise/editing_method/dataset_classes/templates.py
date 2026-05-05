from collections import namedtuple

########
# llama 3
########
def get_llama_with_answer(que, ans):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_llama_without_answer_cot(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a multi-hop explanation for the next question: {que}<|eot_id|>"""

def get_list_llama_without_answer(que, cot):
    if cot == False:
        L = [get_llama_without_answer(line) for line in que]
    else:
        L = [get_llama_without_answer_cot(line) for line in que]
    return L

########
# DeSTA 2.5
########
def get_desta_with_answer(que, ans):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_desta_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
def get_list_desta_without_answer(que):
    return [get_desta_without_answer(line) for line in que]

########
# qwen 2.5
########

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_list_qwen_without_answer(que, cot):
    if cot == False:
        L = [get_qwen_without_answer(line) for line in que]
    else:
        L = [get_qwen_without_answer_cot(line) for line in que]
    return L


Template = namedtuple("Template", ["wo_answer", "wo_answer_list"])

TEMPLATE_DICT = {
    "Llama3-8B-Instruct": Template(get_llama_without_answer, get_list_llama_without_answer),
    "Qwen2-7B-Instruct": Template(get_qwen_without_answer, get_list_qwen_without_answer),
    "DeSTA25-Audio-Llama-3.1-8B": Template(get_desta_without_answer, get_list_desta_without_answer),
    "audio-flamingo-3-hf": Template(get_qwen_without_answer, get_list_qwen_without_answer),  # AF3 uses Qwen2 backbone
}

EOS_DICT = {
    "Llama3-8B-Instruct": "<|eot_id|>",
    "Qwen2-7B-Instruct": "<|im_end|>",
    "DeSTA25-Audio-Llama-3.1-8B": "<|eot_id|>",
    "audio-flamingo-3-hf": "<|im_end|>",  # AF3 uses Qwen2 backbone
}