import os, sys
sys.path.append(os.path.normpath(f'{__file__}/../../../../'))

from packages.chinatsu.main import MnaCircuit

async def hello():
    return { 'message': 'hi' }
