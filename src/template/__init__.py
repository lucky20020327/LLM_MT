import os

_pwd = os.path.dirname(os.path.abspath(__file__))

function_mr_prompt = open(
    os.path.join(_pwd, "function_mr.prompt"), "r", encoding="utf-8"
).read()
function_source_input_generator_prompt = open(
    os.path.join(_pwd, "function_source_input_generator.prompt"),
    "r",
    encoding="utf-8",
).read()
function_followup_input_generator_prompt = open(
    os.path.join(_pwd, "function_followup_input_generator.prompt"),
    "r",
    encoding="utf-8",
).read()
function_valid_code_prompt = open(
    os.path.join(_pwd, "function_valid_code.prompt"),
    "r",
    encoding="utf-8",
).read()
# This is used for functions are not in a public package. So the source_code of the function is needed to be imported.
local_function_test_program_template = open(
    os.path.join(_pwd, "local_function_test_program.template"),
    "r",
    encoding="utf-8",
).read()
