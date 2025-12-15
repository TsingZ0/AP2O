import re


# SYSTEM_PROMPT = """<|im_start|>system
# Please provide a self-contained Python script that solves the following problem in a markdown code block.
# * Consider edge cases and error handling.
# * Keep the function name and parameter list in the answer as described in the problem.
# * Remove examples from the answer.
# <|im_end|>
# <|im_start|>user
# ## Problem:
# {}
# <|im_end|>
# <|im_start|>assistant
# ## Answer:
# """

SYSTEM_PROMPT = "Please provide a Python script that solves the following problem. \n\n{}\n\n```python\n"
# SYSTEM_PROMPT = "Please review and edit the provided Python file according to the requirements, and return the entire Python file. \n\n{}\n\n```python\n"


EOS = [ # add pad_token
    "<|endoftext|>",
    "<|endofmask|>",
    "<｜end▁of▁sentence｜>",
    "<[PAD▁TOKEN]>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\n#",
    "\n```",
]


ENTRANCE_POINT = "solver"


def get_def_line(text):
    for line in text.splitlines():
        if line.strip().startswith("def "):
            return line.strip()
    return None


def get_entrance_point(text):
    matches = re.findall(r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*', text, re.MULTILINE)
    if matches:
        return matches[-1]  # Return the last matched function name
    return None


def post_process(answer, replace_entrance_point=False):
    for eos in EOS:
        if answer.find(eos) > 0:
            answer = answer[:answer.find(eos)]
    if replace_entrance_point:
        answer_entrance_point = get_entrance_point(answer)
        if answer_entrance_point is not None and ENTRANCE_POINT not in answer:
            answer = answer.replace(answer_entrance_point, ENTRANCE_POINT)
    return answer


PASS_SCALAR = 0.0
