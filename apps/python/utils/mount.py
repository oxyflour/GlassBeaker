import os
import importlib.util

import inspect
from fastapi import FastAPI

api_root = os.path.normpath(f'{__file__}/../../')

def mount_module(app: FastAPI, sub_path: str, abs_path: str):
    filename = os.path.basename(abs_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(filename, abs_path)
    module = spec and importlib.util.module_from_spec(spec)
    if module and spec and spec.loader:
        spec.loader.exec_module(module)
        for name, member in inspect.getmembers(module):
            if inspect.iscoroutinefunction(member):
                prefix = f'/{sub_path}/{filename}/{name}'
                print(f'INFO: adding api {prefix}')
                app.add_api_route(prefix, endpoint=member, methods=["GET"])
    else:
        print(f'WARN: load from {abs_path} failed')

def mount_routes(api: FastAPI, sub_path: str):
    path = os.path.join(api_root, sub_path)
    for item in os.listdir(path):
        abs_path = os.path.join(path, item)
        if os.path.isdir(abs_path):
            mount_routes(api, abs_path)
        elif abs_path.endswith('.py'):
            mount_module(api, sub_path, abs_path)
