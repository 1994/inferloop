"""Find files under a crate's `src/` that are only reachable through `#[cfg(test)]`.

Unit test bodies live next to the module they test and are mounted with a
`#[cfg(test)] mod ...;` declaration, so their file names need not carry a
`tests` suffix. Source-policy checks use this to treat such files as test code
rather than production code, the same way they treated `tests/unit/` before the
bodies moved into the module tree.
"""

import re
from pathlib import Path

DECLARATION = re.compile(
    r"(?P<attrs>(?:[ \t]*#\[[^\n]*\]\n)*)[ \t]*(?:pub(?:\([^)]*\))?\s+)?"
    r"mod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*;"
)
PATH_ATTR = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')
TEST_ONLY = re.compile(r"#\[\s*cfg\s*\(\s*(?:all\s*\(\s*)?test\b")


def module_dir(source):
    """The directory Rust searches for a file module's children."""
    if source.name in ("mod.rs", "lib.rs", "main.rs"):
        return source.parent
    return source.parent / source.stem


def declarations(path):
    """Yield `(attributes, target)` for each `mod` declaration in `path`."""
    for match in DECLARATION.finditer(path.read_text(errors="ignore")):
        attrs = match.group("attrs")
        targets = PATH_ATTR.findall(attrs)
        if targets:
            target = (path.parent / targets[-1]).resolve()
        else:
            base = module_dir(path)
            target = (base / f"{match.group('name')}.rs").resolve()
            if not target.is_file():
                alternate = (base / match.group("name") / "mod.rs").resolve()
                if alternate.is_file():
                    target = alternate
        yield attrs, target


def test_source_paths(root):
    """Every resolved file that is only reachable through `#[cfg(test)]`."""
    root = Path(root)
    sources = set()
    pending = []
    # The testing group holds host-only doubles and acceptance cases that are dependencies of
    # test targets, never production builds.
    for path in sorted(root.glob("crates/testing/*/*/src/**/*.rs")):
        sources.add(path.resolve())
    for path in sorted(root.glob("crates/*/*/src/**/*.rs")):
        for attrs, target in declarations(path):
            if TEST_ONLY.search(attrs) and target.is_file():
                sources.add(target)
                pending.append(target)
    while pending:
        parent = pending.pop()
        for _attrs, target in declarations(parent):
            if target.is_file() and target not in sources:
                sources.add(target)
                pending.append(target)
    return sources


def is_test_source(path, sources):
    return Path(path).resolve() in sources
