import inspect
import types
import json
import os


def get_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "(Signature not available)"


def get_source(obj):
    try:
        return inspect.getsource(obj)
    except Exception:
        return "(Source not available)"


def get_doc(obj):
    doc = inspect.getdoc(obj)
    return doc if doc else None


def is_defined_in_package(obj, now_module_name):
    module_name = getattr(obj, "__module__", "") or ""
    # print(f"Module name: {module_name}")
    # print(getattr(obj, "__module__", ""))
    return module_name == now_module_name


def function_filter(obj, name, module_name):
    """Check if the object is a function that we want to include."""
    if not inspect.isfunction(obj):
        return False
    # Exclude abstract methods and methods defined in other modules
    if getattr(obj, "__isabstractmethod__", False):
        return False
    if not is_defined_in_package(obj, module_name):
        return False
    # Exclude functions without docstrings
    if get_doc(obj) is None:
        return False
    # Exclude functions that are possible wrappers or decorators
    try:
        source = inspect.getsource(obj)
        if "wrapper" in source or "decorator" in source:
            return False
    except Exception:
        pass
    if name == "main":
        return False

    # If all checks passed, include the function
    return True


def list_package_api(module, module_name, prefix=""):
    print(f"Listing API for {module_name} with prefix {prefix}")
    api_info = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue  # Skip private/internal names

        full_name = f"{prefix}{name}"

        # Recurse into submodules
        if inspect.ismodule(obj) and getattr(obj, "__name__", "").startswith(
            module_name
        ):
            # list_package_api(obj, prefix=f"{full_name}.")
            pass

        # Handle classes
        elif inspect.isclass(obj) and is_defined_in_package(obj, module_name):
            class_data = {
                "type": "class",
                "name": full_name,
                "signature": get_signature(obj),
                "docstring": get_doc(obj),
                "source_code": get_source(obj),
                "methods": [],
            }

            # Collect methods
            try:
                for method_name, method in inspect.getmembers(obj):
                    if method_name.startswith("_"):
                        continue
                    if function_filter(method, method_name, module_name):
                        method_data = {
                            "name": f"{full_name}.{method_name}",
                            "signature": get_signature(method),
                            "docstring": get_doc(method),
                            "source_code": get_source(method),
                        }
                        class_data["methods"].append(method_data)
            except Exception as e:
                print(f"Error processing class {full_name}: {e}")
                continue
            if class_data["methods"] == []:
                continue  # Skip classes without methods
            api_info.append(class_data)

        # Handle standalone functions
        elif function_filter(obj, name, module_name):
            func_data = {
                "type": "function",
                "name": full_name,
                "signature": get_signature(obj),
                "docstring": get_doc(obj),
                "source_code": get_source(obj),
                "methods": [],  # empty for functions
            }
            api_info.append(func_data)

    return api_info


def list_package_api_traverse(package_name: str):

    package_obj = __import__(package_name, fromlist=[""])
    package_dir = os.path.dirname(package_obj.__file__)
    modules_name = [
        name[:-3]
        for name in os.listdir(package_dir)
        if not name.startswith("_")
        and name.endswith(".py")
        and name != "__init__.py"
        and os.path.isfile(os.path.join(package_dir, name))
    ]
    sub_packages = [
        name
        for name in os.listdir(package_dir)
        if os.path.isdir(os.path.join(package_dir, name))
        and not name.startswith("_")
        and name != "__pycache__"
        and "__init__.py" in os.listdir(os.path.join(package_dir, name))
    ]
    print(f"Modules inside {package_dir}: {modules_name}")
    print(f"Sub-packages inside {package_dir}: {sub_packages}")

    api_info = list_package_api(package_obj, package_name, prefix=f"{package_name}.")

    print(f"API information for {package_name} collected.")
    if api_info != []:
        # Save the API information to a JSON file
        with open(f"{package_name}.json", "w", encoding="utf-8") as f:
            json.dump(api_info, f, indent=2, ensure_ascii=False)

    for module_name in modules_name:
        module_obj = __import__(f"{package_name}.{module_name}", fromlist=[""])
        api_info = list_package_api(
            module_obj,
            f"{package_name}.{module_name}",
            prefix=f"{package_name}.{module_name}.",
        )
        print(f"API information for {package_name}.{module_name} collected.")
        if api_info != []:
            # Save the API information to a JSON file
            with open(f"{package_name}.{module_name}.json", "w", encoding="utf-8") as f:
                json.dump(api_info, f, indent=2, ensure_ascii=False)

    for sub_package in sub_packages:
        list_package_api_traverse(f"{package_name}.{sub_package}")


if __name__ == "__main__":

    package_name = (
        "betterproto"  # betterproto alpyne wenv zugbruecke typer polars river rio astropy
    )

    list_package_api_traverse(package_name)

    # import inspect
    # import abc
    # from abc import ABC, abstractmethod

    # def is_abstract_function(obj):
    #     return inspect.isfunction(obj) and getattr(obj, "__isabstractmethod__", False)
    # def is_property(obj):
    #     return isinstance(obj, property)
    # # Example usage
    # class MyBase(ABC):
    #     @property
    #     def my_property(self):
    #         """This is a property."""
    #         return "property value"
    #     @abc.abstractmethod
    #     def my_method(self):
    #         pass
    # print(inspect.isfunction(MyBase.my_property))
    # print(is_property(MyBase.__dict__['my_property']))  # True
