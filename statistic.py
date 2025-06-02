import json
import os

from loguru import logger


api_file = "/Users/lucky/work/ZJU/2025_04_23_metamorphic_testing/LLM_based_MT/dataset/humaneval/humaneval_mutated.json"
api_infos = json.load(open(api_file, "r", encoding="utf-8"))

logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

valid_mr_count = 0
total_mr_count = 0

mutant_detected_count = 0
total_mutant_count = 0
mutant_detection_statistics = {}

for api_info in api_infos:

    function_full_name = api_info["name"]
    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    mr_evaluate_results_file_path = os.path.join(
        ".",
        "simple_output",
        "mr_evaluate_results",
        module_name.replace(".", os.sep),
        f"{function_name}_mr_evaluate_results.json",
    )
    try:
        mr_evalute_results = json.load(
            open(mr_evaluate_results_file_path, "r", encoding="utf-8")
        )
    except FileNotFoundError:
        logger.warning(
            f"MR evaluate results file not found: {mr_evaluate_results_file_path}"
        )
        continue
    for mr_id, mr_e_result in mr_evalute_results.items():
        if mr_e_result["valid_mr"]:
            valid_mr_count += 1
        total_mr_count += 1

    for mutant in api_info["mutations"]:
        mutant_detection_statistics[mutant["name"]] = {
            "effective_mr_count": 0,
            "total_mr_count": len(mr_evalute_results),
        }
        for mr_id, mr_e_result in mr_evalute_results.items():
            if mr_e_result["valid_mr"] and mr_e_result["mutant_detection_results"].get(
                mutant["name"], False
            ):
                mutant_detection_statistics[mutant["name"]]["effective_mr_count"] += 1

        mutant_detected_count += (
            1
            if mutant_detection_statistics[mutant["name"]]["effective_mr_count"] > 0
            else 0
        )
        total_mutant_count += 1

logger.info(f"Valid MR count: {valid_mr_count}, Total MR count: {total_mr_count}")
logger.info(f"Valid MR rate: {valid_mr_count / total_mr_count:.2%}")

logger.info(
    f"Mutant detected count: {mutant_detected_count}, Total mutant count: {total_mutant_count}"
)
logger.info(f"Mutant detection rate: {mutant_detected_count / total_mutant_count:.2%}")

with open(
    os.path.join(
        ".",
        "simple_output",
        "mr_evaluate_results",
        module_name.split(".")[0],  # the package name
        "mutant_detection_statistics.json",
    ),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        mutant_detection_statistics,
        f,
        indent=4,
        ensure_ascii=False,
    )
logger.info(f"Mutant detection statistics saved to {f.name}")
