import ast
import inspect
import sys
import json
import types
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from loguru import logger


class DeepAPIInspector:
    def __init__(self, api_function, source_code: str):
        """
        初始化API深度检查器

        :param api_function: 要分析的目标API函数
        :param source_code: 函数的源代码字符串
        """
        self.api_function = api_function
        self.function_name = api_function.__name__
        self.function_source = source_code
        try:
            self.ast_tree = ast.parse(self.function_source)
        except SyntaxError as e:
            self.ast_tree = None
            self.parse_error = f"Syntax error at line {e.lineno}: {e.msg}"

        self.control_flow_info = {}
        self.parameter_relations = defaultdict(list)
        self.state_mutations = {
            "parameter_mutations": {},
            "global_access": [],
            "side_effects": [],
        }
        self.computational_properties = {
            "idempotent": False,
            "commutative": False,
            "associative": False,
        }
        self.parameter_sensitivity = {}

    def analyze_control_flow(self) -> Dict[str, int]:
        """
        分析函数的控制流图特征
        """
        if self.ast_tree is None:
            return {"error": self.parse_error}

        class ControlFlowVisitor(ast.NodeVisitor):
            def __init__(self):
                self.stats = {
                    "if_count": 0,
                    "loop_count": 0,
                    "try_count": 0,
                    "with_count": 0,
                    "max_nesting": 0,
                    "return_count": 0,
                }
                self.current_nesting = 0

            def visit_If(self, node):
                self.stats["if_count"] += 1
                self.current_nesting += 1
                self.stats["max_nesting"] = max(
                    self.stats["max_nesting"], self.current_nesting
                )
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_For(self, node):
                self.stats["loop_count"] += 1
                self.current_nesting += 1
                self.stats["max_nesting"] = max(
                    self.stats["max_nesting"], self.current_nesting
                )
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_While(self, node):
                self.stats["loop_count"] += 1
                self.current_nesting += 1
                self.stats["max_nesting"] = max(
                    self.stats["max_nesting"], self.current_nesting
                )
                self.generic_visit(node)
                self.current_nesting -= 1

            def visit_Try(self, node):
                self.stats["try_count"] += 1
                self.generic_visit(node)

            def visit_With(self, node):
                self.stats["with_count"] += 1
                self.generic_visit(node)

            def visit_Return(self, node):
                self.stats["return_count"] += 1
                self.generic_visit(node)

        visitor = ControlFlowVisitor()
        try:
            visitor.visit(self.ast_tree)
            self.control_flow_info = visitor.stats
            return visitor.stats
        except Exception as e:
            return {"error": f"Control flow analysis failed: {str(e)}"}

    def analyze_parameter_usage(self) -> Dict[str, List[str]]:
        """
        分析参数在函数中的使用方式，识别参数之间的依赖关系
        """
        if self.ast_tree is None:
            return {}

        class ParameterVisitor(ast.NodeVisitor):
            def __init__(self, param_names):
                self.param_names = param_names
                self.relations = defaultdict(list)

            def visit_FunctionDef(self, node):
                self.generic_visit(node)

            def visit_If(self, node):
                self._analyze_condition(node.test)
                self.generic_visit(node)

            def visit_For(self, node):
                self._analyze_expression(node.iter)
                self.generic_visit(node)

            def visit_BinOp(self, node):
                left_vars = self._extract_variables(node.left)
                right_vars = self._extract_variables(node.right)
                for var1 in left_vars:
                    for var2 in right_vars:
                        if var1 in self.param_names and var2 in self.param_names:
                            self._add_relation(var1, var2, "binary_operation")
                self.generic_visit(node)

            def _analyze_condition(self, node):
                variables = self._extract_variables(node)
                param_vars = [v for v in variables if v in self.param_names]

                for i in range(len(param_vars)):
                    for j in range(i + 1, len(param_vars)):
                        self._add_relation(param_vars[i], param_vars[j], "condition")

            def _analyze_expression(self, node):
                variables = self._extract_variables(node)
                param_vars = [v for v in variables if v in self.param_names]

                for i in range(len(param_vars)):
                    for j in range(i + 1, len(param_vars)):
                        self._add_relation(param_vars[i], param_vars[j], "expression")

            def _extract_variables(self, node) -> List[str]:
                """递归提取表达式中的变量名"""
                variables = []

                if isinstance(node, ast.Name):
                    variables.append(node.id)
                elif isinstance(node, ast.Attribute):
                    variables.append(node.attr)
                elif isinstance(node, ast.Subscript):
                    variables.extend(self._extract_variables(node.value))
                    if isinstance(node.slice, ast.Index):
                        variables.extend(self._extract_variables(node.slice.value))
                    else:
                        variables.extend(self._extract_variables(node.slice))
                elif hasattr(node, "left") and hasattr(node, "right"):
                    variables.extend(self._extract_variables(node.left))
                    variables.extend(self._extract_variables(node.right))
                elif hasattr(node, "elts"):
                    for elt in node.elts:
                        variables.extend(self._extract_variables(elt))
                elif hasattr(node, "values"):
                    for value in node.values:
                        variables.extend(self._extract_variables(value))
                elif hasattr(node, "operand"):
                    variables.extend(self._extract_variables(node.operand))

                return list(set(variables))

            def _add_relation(self, param1, param2, context):
                if param1 != param2:
                    key = tuple(sorted([param1, param2]))
                    if context not in self.relations[key]:
                        self.relations[key].append(context)

        for node in ast.walk(self.ast_tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.function_name:
                param_names = [arg.arg for arg in node.args.args]
                break
        else:
            sig = inspect.signature(self.api_function)
            param_names = list(sig.parameters.keys())

        if not param_names:
            return {}

        visitor = ParameterVisitor(param_names)
        try:
            visitor.visit(self.ast_tree)
        except Exception as e:
            return {"error": f"Parameter analysis failed: {str(e)}"}

        for (param1, param2), contexts in visitor.relations.items():
            self.parameter_relations[f"{param1}-{param2}"] = {
                "params": [param1, param2],
                "relation_type": contexts,
                "strength": "strong" if "condition" in contexts else "weak",
            }

        return self.parameter_relations

    def analyze_state_mutations(self) -> Dict[str, Any]:
        """
        分析状态变更
        """
        if self.ast_tree is None:
            return {}

        class MutationVisitor(ast.NodeVisitor):
            def __init__(self):
                self.mutations = {
                    "parameter_mutations": {},
                    "global_access": [],
                    "side_effects": [],
                }

            def visit_Assign(self, node):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id in self.mutations["parameter_mutations"]
                    ):
                        self.mutations["parameter_mutations"][target.id].append(
                            {"line": node.lineno, "operation": "assignment"}
                        )
                self.generic_visit(node)

            def visit_AugAssign(self, node):
                if (
                    isinstance(node.target, ast.Name)
                    and node.target.id in self.mutations["parameter_mutations"]
                ):
                    self.mutations["parameter_mutations"][node.target.id].append(
                        {
                            "line": node.lineno,
                            "operation": "augmented_assignment",
                            "operator": type(node.op).__name__,
                        }
                    )
                self.generic_visit(node)

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in [
                        "open",
                        "write",
                        "print",
                        "connect",
                        "send",
                        "input",
                    ]:
                        self.mutations["side_effects"].append(
                            {"function": func_name, "line": node.lineno}
                        )
                self.generic_visit(node)

            def visit_Global(self, node):
                for name in node.names:
                    if name not in self.mutations["global_access"]:
                        self.mutations["global_access"].append(name)
                self.generic_visit(node)

        sig = inspect.signature(self.api_function)
        for name, param in sig.parameters.items():
            if name not in self.state_mutations["parameter_mutations"]:
                self.state_mutations["parameter_mutations"][name] = []

        visitor = MutationVisitor()
        try:
            visitor.visit(self.ast_tree)
            self.state_mutations = visitor.mutations
        except Exception as e:
            self.state_mutations = {"error": f"Mutation analysis failed: {str(e)}"}
        return self.state_mutations

    def analyze_computational_properties(self) -> Dict[str, bool]:
        """
        分析计算属性（启发式方法）
        """
        if self.ast_tree is None:
            return {}

        sig = inspect.signature(self.api_function)
        positional_params = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind == p.POSITIONAL_OR_KEYWORD
        ]

        if len(positional_params) >= 2:
            has_commutative_op = False
            has_associative_op = False

            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.BinOp):
                    if isinstance(node.op, (ast.Add, ast.Mult)):
                        has_commutative_op = True
                        has_associative_op = True
                    elif isinstance(node.op, (ast.Sub, ast.Div)):
                        has_commutative_op = False
                        break

            self.computational_properties["commutative"] = has_commutative_op
            self.computational_properties["associative"] = has_associative_op

            has_random = False
            has_side_effects = bool(
                self.state_mutations and self.state_mutations.get("side_effects")
            )

            for node in ast.walk(self.ast_tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in [
                        "random",
                        "randint",
                        "choice",
                        "time",
                        "datetime",
                    ]:
                        has_random = True
                        break

            self.computational_properties["idempotent"] = (
                not has_random and not has_side_effects
            )

        return self.computational_properties

    def trace_parameter_sensitivity(self, sample_values: dict = None) -> Dict[str, str]:
        """
        动态追踪参数敏感性（边界值分析）

        :param sample_values: 可选的测试值字典，用于不同场景
        """
        if sample_values is None:
            sample_values = {
                "boundary": None,
                "invalid_type": None,
                "large_value": None,
                "small_value": None,
            }

        sensitivity = {}
        sig = inspect.signature(self.api_function)

        for param_name, param in sig.parameters.items():
            param_sens = {
                "boundary_triggered": False,
                "invalid_triggered": False,
                "large_value_issue": False,
            }
            annotations = []

            if param.annotation != inspect.Parameter.empty:
                anno_str = str(param.annotation)
                try:
                    if "int" in anno_str or "float" in anno_str:
                        sample_values["boundary"] = 0
                        sample_values["large_value"] = 10**6
                        sample_values["small_value"] = -(10**6)
                        annotations.append("numeric")

                    if "str" in anno_str:
                        sample_values["boundary"] = ""
                        sample_values["large_value"] = "a" * 1000
                        annotations.append("string")

                    if "List" in anno_str or "Sequence" in anno_str:
                        sample_values["boundary"] = []
                        sample_values["large_value"] = [0] * 100
                        annotations.append("collection")
                except:
                    pass

            if sample_values["boundary"] is not None:
                try:
                    test_args = {name: None for name in sig.parameters}
                    test_args[param_name] = sample_values["boundary"]

                    self.api_function(**test_args)
                except Exception as e:
                    param_sens["boundary_triggered"] = True
                    annotations.append(f"boundary_error:{type(e).__name__}")

            sensitivity[param_name] = {
                "sensitivity_level": "high" if any(param_sens.values()) else "low",
                "sensitive_contexts": annotations,
            }

        self.parameter_sensitivity = sensitivity
        return sensitivity

    def get_ast_summary(self) -> List[str]:
        """
        生成AST的关键路径摘要
        """
        summary = []

        class SummaryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.lines = []

            def visit_If(self, node):
                condition = ast.unparse(node.test).replace("\n", " ")
                self.lines.append(f"IF [{condition}] THEN")
                self.generic_visit(node)
                if node.orelse:
                    self.lines.append("ELSE")
                self.lines.append("END IF")

            def visit_For(self, node):
                target = ast.unparse(node.target)
                iter_source = ast.unparse(node.iter).replace("\n", " ")
                self.lines.append(f"FOR {target} IN {iter_source}")
                self.generic_visit(node)
                self.lines.append("END FOR")

            def visit_While(self, node):
                condition = ast.unparse(node.test).replace("\n", " ")
                self.lines.append(f"WHILE [{condition}] DO")
                self.generic_visit(node)
                self.lines.append("END WHILE")

            def visit_Try(self, node):
                self.lines.append("TRY")
                self.generic_visit(node)

                for handler in node.handlers:
                    if handler.type:
                        ex_type = ast.unparse(handler.type)
                    else:
                        ex_type = "all exceptions"
                    self.lines.append(f"CATCH {ex_type}")
                    self.generic_visit(handler)
                    self.lines.append("END CATCH")

                if node.finalbody:
                    self.lines.append("FINALLY")
                    self.generic_visit(node.finalbody)
                    self.lines.append("END FINALLY")

                self.lines.append("END TRY")

            def visit_Return(self, node):
                value = ast.unparse(node.value).replace("\n", " ") if node.value else ""
                self.lines.append(f"RETURN {value}")

            def visit_Assign(self, node):
                targets = ", ".join(
                    ast.unparse(t).replace("\n", " ") for t in node.targets
                )
                value = ast.unparse(node.value).replace("\n", " ")
                self.lines.append(f"ASSIGN {targets} = {value}")

            def visit_Call(self, node):
                func = ast.unparse(node.func).replace("\n", " ")
                args = ", ".join(
                    ast.unparse(arg).replace("\n", " ") for arg in node.args
                )
                self.lines.append(f"CALL {func}({args})")

        visitor = SummaryVisitor()
        for node in self.ast_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == self.function_name:
                visitor.visit(node)
                break

        return visitor.lines

    def get_full_report(self) -> dict:
        """
        获取完整的分析报告
        """
        self.analyze_control_flow()
        self.analyze_parameter_usage()
        self.analyze_state_mutations()
        self.analyze_computational_properties()
        self.trace_parameter_sensitivity()

        return {
            "function_name": self.function_name,
            "control_flow": self.control_flow_info,
            "parameter_relations": dict(self.parameter_relations),
            "state_mutations": self.state_mutations,
            "computational_properties": self.computational_properties,
            "parameter_sensitivity": self.parameter_sensitivity,
            "ast_summary": self.get_ast_summary(),
        }


def _load_function_from_source(source_code: str, func_name: str):
    """从源代码字符串中动态加载函数"""
    module = types.ModuleType("dynamic_module")
    exec_globals = {
        "__builtins__": __builtins__,
        "List": List,
        "Dict": Dict,
        "Tuple": Tuple,
        "Any": Any,
        "Optional": Optional,
        "float": float,
        "int": int,
        "str": str,
    }

    try:
        exec(source_code, exec_globals)
        func = exec_globals.get(func_name)
        if func is None:
            parts = func_name.split(".")
            current = exec_globals
            for part in parts:
                current = getattr(current, part, None)
                if current is None:
                    return None
            func = current

        return func
    except Exception as e:
        print(f"Error loading function {func_name}: {str(e)}")
        return None


def process_api_file(input_file: str, output_file: str):
    """处理JSON文件中的所有API"""
    with open(input_file, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {str(e)}")
            return

    if isinstance(data, dict):
        data = [data]

    for api in data:
        if "source_code" not in api or "name" not in api:
            print("Skipping API without source_code or name")
            continue

        func_name = api["name"].split(".")[-1]
        source_code = api["source_code"]

        func = _load_function_from_source(source_code, func_name)
        if func is None:
            print(f"Failed to load function: {func_name}")
            api["deep_report"] = {"error": "Failed to load function"}
            continue

        try:
            inspector = DeepAPIInspector(func, source_code)
            report = inspector.get_full_report()
            api["deep_report"] = report
        except Exception as e:
            print(f"Analysis failed for {func_name}: {str(e)}")
            api["deep_report"] = {"error": f"Analysis failed: {str(e)}"}

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)


def process_api(api: dict) -> dict:
    """
    处理单个API字典，返回分析报告
    """
    logger.debug(f"Processing API: {api.get('name', 'unknown')} to get deep report")
    if "deep_report" in api:
        logger.debug(
            f"Deep report already exists, skipping analysis. Deep report: {api['deep_report']}"
        )
        return api
    if "source_code" not in api or "name" not in api:
        raise Exception("API must contain source_code and name")

    func_name = api["name"].split(".")[-1]
    source_code = api["source_code"]

    logger.debug(f"Loading function {func_name} from source code")
    func = _load_function_from_source(source_code, func_name)
    if func is None:
        raise Exception(f"Failed to load function: {func_name}")

    try:
        logger.debug(f"Creating DeepAPIInspector for function {func_name}")
        inspector = DeepAPIInspector(func, source_code)
        logger.debug(f"Running analysis for function {func_name}")
        report = inspector.get_full_report()
    except Exception as e:
        report = {"error": f"Analysis failed: {str(e)}"}

    logger.debug(f"Analysis complete for function {func_name}")
    api["deep_report"] = report
    logger.debug(f"Report for {func_name}: {report}")

    return api


if __name__ == "__main__":
    input_file = "./dataset/humaneval/humaneval_example.json"
    output_file = "./dataset/humaneval/humaneval_example_modified.json"

    print(f"Processing {input_file}...")
    process_api_file(input_file, output_file)
    print(f"Analysis complete! Results saved to {output_file}")
