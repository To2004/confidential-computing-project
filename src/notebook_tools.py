"""Display helpers for the notebooks.

The notebooks import the experiment code from `src/` rather than copying it, so
the numbers they produce come from the same code the report used. The same rule
applies to the code they *show*: `show_source` reads a definition out of the file
on disk, so a snippet in a notebook can never drift from the implementation.

Source is read as text rather than imported, so a definition can still be shown
when the module it lives in needs a library that is unavailable — `openfhe`
imports at module scope in several files and is Linux-only.
"""

import ast
import io
import os

import project_paths

_DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def source_of(filename, name):
    """Return the source text of `name` as defined in `src/<filename>`.

    `name` may be a plain function or class name, or `Class.method`.
    """
    path = os.path.join(project_paths.SRC_DIR, filename)
    text = io.open(path, encoding="utf-8").read()
    lines = text.splitlines()

    parts = name.split(".")
    scope = ast.parse(text)
    node = None
    for part in parts:
        node = next((child for child in ast.iter_child_nodes(scope)
                     if isinstance(child, _DEFINITION_NODES) and child.name == part), None)
        if node is None:
            raise NameError(f"{name!r} not found in {filename} (looking for {part!r})")
        scope = node

    # Decorators sit above the `def` line and are part of the definition.
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return "\n".join(lines[start:node.end_lineno])


def assignment_in(filename, function, target):
    """Return the source of the `target = ...` statement inside `function`.

    Used to show one statement out of a long function — the `operations` dict
    each backend hands to the harness, for instance — without reproducing it.
    """
    path = os.path.join(project_paths.SRC_DIR, filename)
    text = io.open(path, encoding="utf-8").read()
    lines = text.splitlines()

    func = next((node for node in ast.walk(ast.parse(text))
                 if isinstance(node, _DEFINITION_NODES) and node.name == function), None)
    if func is None:
        raise NameError(f"{function!r} not found in {filename}")

    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target for t in node.targets):
            return "\n".join(lines[node.lineno - 1:node.end_lineno])
    raise NameError(f"no assignment to {target!r} inside {function!r} in {filename}")


def show_source(filename, name, note=""):
    """Render a definition from `src/<filename>` as a Markdown code block."""
    code = source_of(filename, name)
    header = f"**`src/{filename}` — `{name}`**"
    if note:
        header += f"  \n{note}"
    _display_markdown(f"{header}\n\n```python\n{code}\n```")


def show_assignment(filename, function, target, note=""):
    """Render one assignment statement from inside a function."""
    code = assignment_in(filename, function, target)
    header = f"**`src/{filename}` — `{target}` in `{function}`**"
    if note:
        header += f"  \n{note}"
    _display_markdown(f"{header}\n\n```python\n{code}\n```")


def show_sources(items):
    """Render several definitions. `items` is a list of (filename, name, note)."""
    for entry in items:
        filename, name = entry[0], entry[1]
        note = entry[2] if len(entry) > 2 else ""
        show_source(filename, name, note)


def show_markdown(text):
    """Render Markdown, falling back to plain text outside IPython."""
    _display_markdown(text)


def _display_markdown(text):
    try:
        from IPython.display import Markdown, display
    except ImportError:
        print(text)
        return
    display(Markdown(text))
