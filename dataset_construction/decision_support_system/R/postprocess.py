import json
import os
import rpy2.robjects
import re

# filter out the items that are not local executable functions or have empty example code

# read the json file rmcda_full_api.json
# for each of the items, load library RMCDA and try to execute the example code

RMCDA_code_template = """library(RMCDA)

# function source_code
{source_code}

# example code
{example_code}"""

processed_RMCDA = []

rmcda_dataset = json.load(open("./rmcda_full_api.json", "r"))
for name, item in rmcda_dataset.items():
    print(f"Processing {item['name']}...")

    item = {
        "type": "local_function",
        **item,
    }  # ensure item is a dict and has the correct type
    item["mutations"] = []

    source_code = item["source_code"]
    example_code = item["example_code"]

    if example_code == {}:
        print(f"Skipping {item['name']} due to empty example code.")
        continue

    code2exec = RMCDA_code_template.format(
        source_code=source_code, example_code=example_code
    )

    # try to execute the code and catch any errors
    try:
        rpy2.robjects.r(code2exec)
    except Exception as e:
        print(f"Error processing {item['name']}: {e}")
        # use re to extract function name from the error message '错误于function_name(parameters): 没有"function_name"这个函数'
        pattern = r"错误于(.*)\(.*\): 没有\"(.*)\"这个函数"
        match = re.search(pattern, str(e))
        if match:
            function_name = match.group(2)
            print(f"Function name extracted: {function_name}")
            print(f"Modifying function name to RMCDA::{function_name} and retrying...")
            code2exec = code2exec.replace(function_name, "RMCDA:::" + function_name)
            item["source_code"] = item["source_code"].replace(
                function_name, "RMCDA:::" + function_name
            )
            item["example_code"] = item["example_code"].replace(
                function_name, "RMCDA:::" + function_name
            )
            try:
                rpy2.robjects.r(code2exec)
            except Exception as e:
                print(
                    f"Error processing {item['name']} with modified function name: {e}"
                )
            else:
                processed_RMCDA.append(item)
    else:
        processed_RMCDA.append(item)

    print(f"Processed {item['name']}")

# save the processed RMCDA dataset to a json file
with open("./rmcda_processed_api.json", "w") as f:
    json.dump(processed_RMCDA, f, indent=4)
