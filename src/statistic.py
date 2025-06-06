import json
import os
import argparse
import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

pwd = os.path.dirname(os.path.abspath(__file__))


def arg_parser():
    parser = argparse.ArgumentParser(description="Simple MR for local functions.")
    parser.add_argument(
        "--api_file",
        type=str,
        required=True,
        help="Path to the API info file.",
    )
    parser.add_argument(
        "--log_base_dir",
        type=str,
        default=os.path.join(pwd, "..", "logs", "simple_mr"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated test programs and metamorphic relations.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        choices=["simple", "func_deep_report"],
        help="Strategy for generating metamorphic relations.",
    )
    return parser


def collect_coverage_data(mr_evaluate_results):
    coverage_data = []
    for mr in mr_evaluate_results.values():
        if mr["valid_mr"] and mr["coverage_info"]:
            branch_total = mr["coverage_info"]["branch_coverage"]["total"]
            has_valid_branch = branch_total > 0

            if has_valid_branch:
                coverage_data.append(
                    {
                        "statement": mr["coverage_info"]["statement_coverage"][
                            "percent"
                        ],
                        "branch": mr["coverage_info"]["branch_coverage"]["percent"],
                    }
                )
            else:
                coverage_data.append(
                    {"statement": mr["coverage_info"]["statement_coverage"]["percent"]}
                )

    return coverage_data


def draw_coverage_boxplot(coverage_results, save_path=None):
    statement_percents = [res["statement"] for res in coverage_results]
    data_to_plot = [statement_percents]
    labels = ["Statement Coverage"]
    palette = ["#2ecc71"]

    if len(coverage_results) > 0 and "branch" in coverage_results[0]:
        branch_percents = [res["branch"] for res in coverage_results]
        data_to_plot.append(branch_percents)
        labels.append("Branch Coverage")
        palette.append("#3498db")

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=data_to_plot, palette=palette, width=0.4)

    plt.title("Code Coverage Distribution", fontsize=14)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel("Coverage Percentage (%)")
    plt.ylim(0, 110)

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    args = arg_parser().parse_args()

    api_file = args.api_file
    api_infos = json.load(open(api_file, "r", encoding="utf-8"))

    logger.remove()
    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(
        os.path.join(
            args.log_base_dir,
            args.strategy,
            "statistic",
            f"{now_time_str}.log",
        ),
        level="DEBUG",
        format="{time} {level} {file}|{line}: {message}",
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time}</green> <level>{level}</level> <cyan>{file}</cyan>|<magenta>{line}</magenta>: <level>{message}</level>",
        colorize=True,
    )
    logger.info(f"Starting mutant detection statistics computation...")
    logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

    valid_mr_count = 0
    total_mr_count = 0

    mutant_detected_count = 0
    total_mutant_count = 0
    mutant_detection_statistics = {}
    coverage_statistics = []

    coverage_plot_dir = os.path.join(args.output_dir, args.strategy, "box_plot")
    os.makedirs(coverage_plot_dir, exist_ok=True)

    for api_info in api_infos:

        function_full_name = api_info["name"]
        function_name = function_full_name.split(".")[-1]
        module_name = ".".join(function_full_name.split(".")[:-1])

        mr_evaluate_results_file_path = os.path.join(
            args.output_dir,
            args.strategy,
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

        coverage_data = collect_coverage_data(mr_evalute_results)
        coverage_plot_path = os.path.join(
            args.output_dir,
            args.strategy,
            "box_plot",
            module_name.replace(".", os.sep),
            f"{function_name}.png",
        )
        coverage_statistics.extend(coverage_data)
        draw_coverage_boxplot(coverage_data, coverage_plot_path)
        logger.info(f"Coverage visualization saved to {coverage_plot_path}")

        for mutant in api_info["mutations"]:
            mutant_detection_statistics[mutant["name"]] = {
                "effective_mr_count": 0,
                "total_mr_count": len(mr_evalute_results),
            }
            for mr_id, mr_e_result in mr_evalute_results.items():
                if mr_e_result["valid_mr"] and mr_e_result[
                    "mutant_detection_results"
                ].get(mutant["name"], False):
                    mutant_detection_statistics[mutant["name"]][
                        "effective_mr_count"
                    ] += 1

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
    logger.info(
        f"Mutant detection rate: {mutant_detected_count / total_mutant_count:.2%}"
    )

    with open(
        os.path.join(
            args.output_dir,
            args.strategy,
            "mr_evaluate_results",
            module_name.split(".")[0],  # the package name
            "aaa_mutant_detection_statistics.json",
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

    coverage_statistics_plot_path = os.path.join(
        args.output_dir,
        args.strategy,
        "box_plot",
        module_name.split(".")[0],  # the package name
        "aaa_coverage_statistics.png",
    )
    draw_coverage_boxplot(coverage_statistics, coverage_statistics_plot_path)
    logger.info(
        f"Coverage statistics visualization saved to {coverage_statistics_plot_path}"
    )
