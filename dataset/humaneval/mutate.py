import ast
import json
import signal

from mutmut.file_mutation import mutate_file_contents


# Define a timeout exception
class TimeoutException(Exception):
    pass


# Define signal handler
def handler(signum, frame):
    raise TimeoutException("Execution timed out!")


with open("./humaneval.json", "r") as f:
    data = json.load(f)
# Process each item in the dataset

for i, item in enumerate(data):
    source_code = item["source_code"]
    test_code = item["test_code"]
    item["mutations"] = []

    # Step 1: Generate mutants
    mutant_code_str, mutant_names = mutate_file_contents(item["name"], source_code)

    # Step 2: Parse original source
    original_ast = ast.parse(source_code)

    # Step 3: Parse full mutant file
    mutant_ast = ast.parse(mutant_code_str)

    # Step 4: Build mapping from mutant name to its AST function node
    mutant_func_map = {}
    for node in mutant_ast.body:
        if (
            isinstance(node, ast.FunctionDef)
            and "__mutmut_" in node.name
            and not node.name.endswith("_orig")
        ):
            # Get original function name
            prefix = node.name.split("__mutmut_")[0]
            orig_func_name = prefix[2:]  # remove "x_"
            mutant_func_map.setdefault(orig_func_name, []).append(node)

    # Step 5: Replace original function body with each mutant and save
    for func_node in original_ast.body:
        if isinstance(func_node, ast.FunctionDef):
            func_name = func_node.name
            if func_name in mutant_func_map:
                for mutant_node in mutant_func_map[func_name]:
                    # Clone original AST
                    mutated_ast = ast.parse(source_code)
                    # Replace body of the corresponding function
                    for m_func in mutated_ast.body:
                        if (
                            isinstance(m_func, ast.FunctionDef)
                            and m_func.name == func_name
                        ):
                            m_func.body = mutant_node.body
                    # Write to file
                    mutated_code = ast.unparse(mutated_ast)
                    mutation_name = f"mutated_{mutant_node.name}"

                    # ensure the mutation can be executed with an assertion error
                    executed_code = f"""
{mutated_code}
{test_code}
"""
                    try:
                        signal.signal(signal.SIGALRM, handler)
                        signal.alarm(2)  # Timeout after 2 seconds

                        exec(executed_code)
                    except AssertionError:
                        print(f"Mutation {mutation_name} failed assertion check.")
                        item["mutations"].append(
                            {
                                "name": mutation_name,
                                "source_code": mutated_code,
                            }
                        )
                    except TimeoutException:
                        print(f"Mutation {mutation_name} timed out.")
                    except Exception as e:
                        print(
                            f"Mutation {mutation_name} raised an unexpected error: {e}"
                        )
                    else:
                        print(f"Mutation {mutation_name} passed assertion check.")
                    finally:
                        signal.alarm(0)

    # Save the mutated item back to the dataset
    with open("./humaneval_mutated.json", "w") as f:
        json.dump(data, f, indent=4)
    print(
        f"Processed item {i + 1}/{len(data)}: {item['name']} with {len(item['mutations'])} mutations"
    )
    print(f"Mutations saved to humaneval_mutated.json")
