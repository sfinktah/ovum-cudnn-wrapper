import importlib
import os
import re
import sys

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
module_root_directory = os.path.dirname(os.path.realpath(__file__))

NODE_CLASS_MAPPINGS = {
}

NODE_DISPLAY_NAME_MAPPINGS = {
}


# noinspection PyShadowingNames
def pretty(name:str):
    return " ".join(re.findall("[A-Z]*[a-z]*", name))

for module in [os.path.splitext(f)[0] for f in os.listdir(module_root_directory) if f.endswith('.py') and not f.startswith('_')]:
    imported_module = importlib.import_module(f"{module}")
    # Legacy pattern: modules export a list 'CLAZZES' of node classes
    if 'CLAZZES' in imported_module.__dict__:
        for clazz in imported_module.CLAZZES:
            name = clazz.__name__
            NODE_CLASS_MAPPINGS[name] = clazz
            display_name = getattr(clazz, "NAME", None)
            if isinstance(display_name, str) and display_name.strip():
                NODE_DISPLAY_NAME_MAPPINGS[name] = display_name
            else:
                NODE_DISPLAY_NAME_MAPPINGS[name] = pretty(name)
    # Autonode pattern: modules export CLASS_MAPPINGS and CLASS_NAMES
    elif 'CLASS_MAPPINGS' in imported_module.__dict__ and 'CLASS_NAMES' in imported_module.__dict__:
        for name, clazz in imported_module.CLASS_MAPPINGS.items():
            NODE_CLASS_MAPPINGS[name] = clazz
        for name, display_name in imported_module.CLASS_NAMES.items():
            # Use provided custom name if present, otherwise pretty name
            NODE_DISPLAY_NAME_MAPPINGS[name] = display_name or pretty(name)

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Ensure web routes in _mini_webserver are registered on package import
try:
    from . import _cudnn_wrapper_api  # noqa: F401
except Exception:
    # Do not fail package import if optional server components are unavailable
    pass


