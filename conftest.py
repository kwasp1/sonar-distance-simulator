"""Makes the repo root importable so tests can `from src... import ...`.

pytest inserts the rootdir into sys.path when a conftest.py lives there,
so `pytest` works the same as `python -m pytest` from any directory.
"""
