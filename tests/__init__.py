"""Makes `tests` a real package.

Two test modules import a shared helper via a package-qualified path
(`from tests.server_fakes import ...`, `from tests.test_orchestrator import
...`). Without this file, pytest's "rootless" import mode inserts only the
`tests/` directory itself onto `sys.path` for each collected file — enough to
import a test module by its bare name, but not enough to resolve `tests.xxx`,
since `tests`' own parent (the repo root) was never added.

With this file present, pytest walks up from each test file through
directories containing `__init__.py` and inserts the first ancestor that does
NOT have one — the repo root — onto `sys.path`. That makes `tests.*` imports
resolve identically under `pytest`, `python -m pytest`, any working directory,
and any OS. (The prior fragile behavior worked only under `python -m pytest`,
which separately prepends the current directory to `sys.path` — that is a
property of `-m`, not of pytest, and does not hold for the plain `pytest`
console script CI invokes.)
"""
