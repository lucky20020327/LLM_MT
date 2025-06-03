import json
import ast
import inspect
import subprocess
import os

from bigcodebench.data.bigcodebench import get_bigcodebench


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


import ast

# (1) Define your sets of “risky” names/modules/calls:
DANGEROUS_BUILTINS = {"eval", "exec", "compile"}
DANGEROUS_MODULES = {
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "urllib",
    "requests",
    "matplotlib",
}
DANGEROUS_CALLS = {
    # (module, attribute) pairs to flag
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rmdir"),
    ("os", "mkdir"),
    ("os", "makedirs"),
    ("shutil", "rmtree"),
    ("os", "system"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "run"),
    ("socket", "socket"),
    ("socket", "create_connection"),
    ("urllib.request", "urlopen"),
    ("requests", "get"),
    ("requests", "post"),
    ("pathlib", "Path"),  # Path(...).write_text or Path(...).unlink
}


def _get_full_attr_name(node: ast.Attribute) -> str:
    """
    Recursively reconstructs things like `urllib.request.urlopen`
    from an ast.Attribute chain.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    parts.reverse()
    return ".".join(parts)


def find_dangerous_nodes(code_str: str):
    """
    Parse `code_str` and return a list of “flags” indicating where
    we saw a dangerous import or call.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        # If code_str isn’t even valid Python, bail out or decide how you want to handle it.
        raise

    flags = []  # list of (lineno, description) for anything dangerous

    for node in ast.walk(tree):
        # 1. Look for "import X" or "from X import Y"
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name  # e.g. "os", "requests", etc.
                # If the top‐level module is in DANGEROUS_MODULES, flag it.
                top = mod_name.split(".")[0]
                if top in DANGEROUS_MODULES:
                    flags.append(
                        (node.lineno, f"import of dangerous module '{mod_name}'")
                    )

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0]
            if top in DANGEROUS_MODULES:
                flags.append((node.lineno, f"from '{mod}' import ..."))

        # 2. Look for calls to builtins like eval/exec
        elif isinstance(node, ast.Name):
            if node.id in DANGEROUS_BUILTINS and isinstance(node.ctx, ast.Load):
                flags.append((node.lineno, f"builtin '{node.id}' used"))

        # 3. Look for function calls: ast.Call
        elif isinstance(node, ast.Call):
            # 3a. If func is Name, check if it’s in dangerous builtins (already done above)
            # 3b. If func is an Attribute, reconstruct full name
            func = node.func
            if isinstance(func, ast.Attribute):
                full = _get_full_attr_name(func)
                # e.g. "os.remove", "subprocess.Popen", "requests.get", etc.
                for mod_name, fn_name in DANGEROUS_CALLS:
                    candidate = f"{mod_name}.{fn_name}"
                    if full == candidate:
                        flags.append((node.lineno, f"call to '{candidate}'"))

            elif isinstance(func, ast.Name):
                # Maybe user did “from os import remove” and then calls remove(...)
                # In that case, node.func.id == "remove". Check if ("os", "remove") is in our set.
                for mod_name, fn_name in DANGEROUS_CALLS:
                    if func.id == fn_name:
                        flags.append(
                            (node.lineno, f"potential call to '{mod_name}.{fn_name}'")
                        )

    return flags


if __name__ == "__main__":

    if not os.path.exists("bigcodebench_data_raw.json"):
        print("Fetching BigCodeBench data...")
        bigcodebench_data = get_bigcodebench()
        print(f"Retrieved {len(bigcodebench_data)} tasks from BigCodeBench.")
        print(f"Example task: {list(bigcodebench_data.items())[0]}")

        for key, item in bigcodebench_data.items():
            item["doc_struct"] = json.loads(item["doc_struct"])
            try:
                item["libs"] = json.loads(item["libs"]) if item.get("libs") else []
            except json.JSONDecodeError:
                print(f"Error decoding libs for task {key}, setting to empty list.")
                item["libs"] = []
            item["libs"].extend(
                [req.split(".")[0] for req in item["doc_struct"].get("reqs", [])]
            )
            item["libs"] = list(set(item["libs"]))  # Remove duplicates

        # save the data to a file
        with open("bigcodebench_data_raw.json", "w") as f:
            json.dump(bigcodebench_data, f, indent=4)
    else:
        print("Loading BigCodeBench data from file...")
        with open("bigcodebench_data_raw.json", "r") as f:
            bigcodebench_data = json.load(f)
        print(f"Loaded {len(bigcodebench_data)} tasks from BigCodeBench data file.")

    formatted_data = []

    for key, item in bigcodebench_data.items():
        func_type = "local_function"
        func_name = f"task_func{key.split('/')[-1]}"

        full_code = item["complete_prompt"] + "\n" + item["canonical_solution"]
        full_code = full_code.replace("task_func", func_name)

        test_code = (
            "import traceback\n"
            + item["test"]
            + "\n"
            + """class RaisingTestResult(unittest.TextTestResult):
    def addError(self, test, err):
        super().addError(test, err)
        print("\\n[addError] Test raised an error:")
        traceback.print_exception(*err)
        raise err[1]  # raise the actual exception

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print("\\n[addFailure] Test failed:")
        traceback.print_exception(*err)
        raise AssertionError(err[1])

class MyTestRunner(unittest.TextTestRunner):
    resultclass = RaisingTestResult
"""
            + "\nif __name__ == '__main__':\n    unittest.main(testRunner=MyTestRunner, exit=False)"
        )
        test_code = test_code.replace("task_func", func_name)

        # libs = item.get("libs", [])
        # # pip install the libraries if they are not already installed
        # for lib in libs:
        #     try:
        #         subprocess.run(["pip", "install", lib], check=True)
        #     except subprocess.CalledProcessError as e:
        #         print(f"Failed to install {lib} for task {key}: {e}")
        #         continue

        # check for dangerous imports or calls
        dangerous_flags = find_dangerous_nodes(full_code)
        dangerous_flags += find_dangerous_nodes(test_code)
        if dangerous_flags:
            print(f"Dangerous code detected in task {key}:")
            for lineno, desc in dangerous_flags:
                print(f"  Line {lineno}: {desc}")
            continue

        # parse full code
        try:
            func_info = extract_function_info(full_code, func_name)
        except Exception as e:
            print(f"Error extracting function info for task {key}: {e}")
            continue

        signature = func_info["signature"]
        docstring = func_info["docstring"]
        methods = []

        formatted_data.append(
            {
                "type": func_type,
                "name": "bigcodebench." + func_name,
                "signature": signature,
                "docstring": docstring,
                "source_code": full_code,
                "test_code": test_code,
                "methods": methods,
            }
        )
    # save the formatted data to a file
    with open("bigcodebench.json", "w") as f:
        json.dump(formatted_data, f, indent=4)
