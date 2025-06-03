import os
import json
import ast

dataset = json.load(open("bigcodebench.json", "r"))


# count the length of the source_code for each item using ast
def count_source_code_length(item):
    try:
        # Parse the source code into an AST
        tree = ast.parse(item["source_code"])
        # Return the number of nodes in the AST
        return len(list(ast.walk(tree)))
    except SyntaxError:
        # If there's a syntax error, return 0
        return 0


# sort the items by the length of the source_code (in ast node level) in descending order
sorted_items = sorted(dataset, key=count_source_code_length, reverse=True)


# take the first 200 items
hardest_items = sorted_items[:100]
# save the hardest items to a new file
with open("bigcodebench_hardest.json", "w") as f:
    json.dump(hardest_items, f, indent=4)
print(f"Saved {len(hardest_items)} hardest items to bigcodebench_hardest.json")
