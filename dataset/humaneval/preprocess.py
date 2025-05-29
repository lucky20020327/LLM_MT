import os
import json
import ast
import inspect


def extract_function_info(code_str, target_function_name):
    # Parse the code string into an AST
    tree = ast.parse(code_str)

    # Find the first function definition in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            if func_name != target_function_name:
                continue

            # If the function name matches, extract its information

            # Reconstruct the function to use inspect
            local_ns = {}
            exec(code_str, {}, local_ns)
            func_obj = local_ns[func_name]

            # Get the signature using inspect
            signature = str(inspect.signature(func_obj))
            docstring = ast.get_docstring(node)

            return {"name": func_name, "signature": signature, "docstring": docstring}

    return None  # No function found


with open("./humaneval_fix.json", "r") as f:
    data = json.load(f)

processed_data = []
names = []

for i, item in enumerate(data):
    source_code = item["oracle_code"]
    entry_point = item["entry_point"]
    function_info = extract_function_info(source_code, entry_point)

    type = "local_function"
    name = "humaneval." + entry_point
    signature = function_info["signature"]
    docstring = function_info["docstring"]
    methods = []

    if name in names:
        print(f"Duplicate function name found: {name} at index {i}")
        continue
    
    processed_data.append(
        {
            "type": type,
            "name": name,
            "signature": signature,
            "docstring": docstring,
            "source_code": source_code,
            "methods": methods,
        }
    )
    names.append(name)
    
with open("./humaneval_processed.json", "w") as f:
    json.dump(processed_data, f, indent=4)
print(f"Processed {len(processed_data)} entries from humaneval_fix.json")
