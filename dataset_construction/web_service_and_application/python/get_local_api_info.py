import inspect
import json
import re
import sys
import importlib
import os
from types import ModuleType, FunctionType, MethodType

def install_package(package_name):
    """安装指定的Python包"""
    try:
        importlib.import_module(package_name)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def get_function_info(full_name, func_obj):
    """获取函数或方法的信息"""
    try:
        try:
            signature = str(inspect.signature(func_obj))
        except ValueError:
            signature = "(...)"
        
        docstring = inspect.getdoc(func_obj) or ""
        
        try:
            source_code = inspect.getsource(func_obj)
        except (TypeError, OSError):
            source_code = f"# Source code not available for {full_name}"
        
        return {
            "package": full_name.split(".")[0],
            "name": full_name,
            "signature": signature,
            "docstring": docstring,
            "source_code": source_code,
        }
    
    except Exception as e:
        print(f"Error processing {full_name}: {str(e)}")
        return None

def extract_class_methods(cls_obj, class_name, package_name, api_dict, visited=None):
    """递归提取类及其嵌套类的方法"""
    if visited is None:
        visited = set()
    
    cls_id = id(cls_obj)
    if cls_id in visited:
        return
    visited.add(cls_id)
    
    for name, member in inspect.getmembers(cls_obj):
        if name.startswith('_'):
            continue
            
        full_member_name = f"{class_name}.{name}"
        
        if inspect.isclass(member):
            nested_class_name = f"{class_name}.{name}"
            extract_class_methods(
                member, 
                nested_class_name, 
                package_name, 
                api_dict, 
                visited
            )
        elif inspect.isfunction(member) or inspect.ismethod(member):
            method_info = get_function_info(full_member_name, member)
            if method_info:
                api_dict[full_member_name] = method_info

def extract_api(package_name="flask"):
    """提取 API 信息（包括递归类方法）"""

    install_package(package_name)
    
    flask = importlib.import_module(package_name)
    
    members = inspect.getmembers(flask)
    
    api_dict = {}
    
    for name, obj in members:
        if name.startswith('_'):
            continue
            
        full_name = f"{package_name}.{name}"
        
        if inspect.isfunction(obj):
            if func_info := get_function_info(full_name, obj):
                api_dict[full_name] = func_info
        
        elif inspect.isclass(obj):
            extract_class_methods(obj, full_name, package_name, api_dict)
    
    return api_dict

def save_to_json(data, package_name):
    """将数据保存为 JSON 文件"""
    filename = f"{package_name}_raw_api.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Exported full API specs to: {os.path.abspath(filename)}")
    return filename

if __name__ == "__main__":
    package = "flask"
    api_data = extract_api(package)
    save_to_json(api_data, package)