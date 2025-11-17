#!/usr/bin/env python3

"""
A self-contained **pure-Python 3.9+** utility for applying human-readable
"pseudo-diff" patch files to a collection of text files.

Source: https://cookbook.openai.com/examples/gpt4-1_prompting_guide

Overview:
---------
This module enables the model to apply code changes by generating a custom patch format
that is more forgiving than traditional unified diff. The patch format supports:
- Adding new files
- Deleting files
- Updating existing files with fuzzy matching
- Moving/renaming files

Patch Format:
-------------
Patches are bounded by *** Begin Patch / *** End Patch markers and contain:
- *** Add File: path/to/new/file.py
  + line 1 of new file
  + line 2 of new file

- *** Delete File: path/to/old/file.py

- *** Update File: path/to/existing/file.py
  @@ optional context line for locating changes
   unchanged line
  -removed line
  +added line
   unchanged line
  *** End of File (optional marker for end-of-file changes)

Fuzzy Matching:
---------------
Unlike strict unified diff, this parser:
1. Tries exact line matching first
2. Falls back to rstrip() matching (ignoring trailing whitespace)
3. Falls back to strip() matching (ignoring all leading/trailing whitespace)
4. Can handle end-of-file markers for changes at file end
This makes it more robust to model-generated patches that might have minor formatting issues.

Integration with Model:
-----------------------
The model learns to generate patches in this format during training. When the model wants
to modify code, it outputs a patch string which is then parsed and applied by this module.
The forgiving nature of the parser helps handle imperfect model outputs.

Safety:
-------
- Validates all file paths before applying changes
- Raises DiffError for any parsing or application issues
- Does not modify files in-place until entire patch is validated
- Supports custom open/write/remove functions for sandboxing
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)


# --------------------------------------------------------------------------- #
#  Domain objects
# --------------------------------------------------------------------------- #
class ActionType(str, Enum):
    """
    The type of change being made to a file.

    - ADD: Creating a new file
    - DELETE: Removing an existing file
    - UPDATE: Modifying an existing file (with optional move/rename)
    """
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    """
    Represents a single file modification in a commit.

    This is the final, validated representation of a change after parsing.
    It contains the actual file contents (before and after) rather than
    just the delta.

    Attributes:
        type: The kind of change (ADD, DELETE, or UPDATE)
        old_content: Original file content (for DELETE and UPDATE)
        new_content: New file content (for ADD and UPDATE)
        move_path: If set, the file should be moved/renamed to this path (UPDATE only)
    """
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    """
    A collection of file changes that should be applied atomically.

    The keys are file paths, and values are FileChange objects describing
    what to do with each file.

    Attributes:
        changes: Dict mapping file paths to their respective changes
    """
    changes: Dict[str, FileChange] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Exceptions
# --------------------------------------------------------------------------- #
class DiffError(ValueError):
    """
    Raised when there's a problem parsing or applying a patch.

    This includes:
    - Malformed patch syntax
    - Missing required files
    - Context lines that can't be found
    - Invalid file operations (e.g., adding a file that already exists)
    """


# --------------------------------------------------------------------------- #
#  Helper dataclasses used while parsing patches
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """
    Represents a contiguous block of changes within a file.

    A chunk describes a location in the original file where lines should be
    removed and/or inserted. The orig_index is determined during parsing by
    finding the context lines in the original file.

    Attributes:
        orig_index: Line number in original file where this chunk applies
        del_lines: Lines to remove from the original file
        ins_lines: Lines to insert in their place
    """
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    """
    Intermediate representation of a file operation during parsing.

    This gets converted to a FileChange after all chunks are applied.

    Attributes:
        type: The kind of change (ADD, DELETE, or UPDATE)
        new_file: Complete content for ADD actions
        chunks: List of Chunk objects for UPDATE actions
        move_path: Target path for move/rename operations
    """
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    """
    Collection of PatchActions parsed from patch text.

    This is an intermediate representation that gets converted to a Commit
    after validation and chunk application.

    Attributes:
        actions: Dict mapping file paths to their respective PatchActions
    """
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  Patch text parser
# --------------------------------------------------------------------------- #
@dataclass
class Parser:
    """
    State machine for parsing patch text into a Patch object.

    The parser maintains an index into the lines of the patch file and
    incrementally builds up PatchActions as it encounters directives like
    "*** Add File" or "*** Update File".

    Fuzzy Matching:
    The parser tracks a "fuzz" score indicating how much fuzzy matching was
    required. Higher fuzz means the patch was less precise:
    - 0: Exact match
    - 1-99: Whitespace differences (rstrip/strip matching)
    - 10000+: End-of-file context found in wrong location

    Attributes:
        current_files: Dict of existing files (path -> content) for validation
        lines: The patch text split into lines
        index: Current line being parsed
        patch: The Patch object being built
        fuzz: Accumulated fuzz score from fuzzy matching
    """
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0

    # ------------- low-level helpers -------------------------------------- #
    def _cur_line(self) -> str:
        """
        Returns the current line being parsed.

        Raises:
            DiffError: If we've reached the end of the patch unexpectedly
        """
        if self.index >= len(self.lines):
            raise DiffError("Unexpected end of input while parsing patch")
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """
        Normalize line endings for cross-platform compatibility.

        Strip CR so comparisons work for both LF and CRLF input.
        """
        return line.rstrip("\r")

    # ------------- scanning convenience ----------------------------------- #
    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if (
            prefixes
            and len(prefixes) > 0
            and self._norm(self._cur_line()).startswith(prefixes)
        ):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        Consume the current line if it starts with *prefix* and return the text
        **after** the prefix.  Raises if prefix is empty.
        """
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """Return the current raw line and advance."""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- public entry point -------------------------------------- #
    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            # ---------- UPDATE ---------- #
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate update for file: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error - missing file: {path}")
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- DELETE ---------- #
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate delete for file: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error - missing file: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- ADD ---------- #
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate add for file: {path}")
                if path in self.current_files:
                    raise DiffError(f"Add File Error - file already exists: {path}")
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise DiffError(f"Unknown line while parsing: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise DiffError("Missing *** End Patch sentinel")
        self.index += 1  # consume sentinel

    # ------------- section parsers ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid line in update section:\n{self._cur_line()}")

            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [
                    s.strip() for s in lines[:index]
                ]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)
            new_index, fuzz = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise DiffError(
                    f"Invalid {'EOF ' if eof else ''}context at {index}:\n{ctx_txt}"
                )
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_ctx)
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done(
            ("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")
        ):
            s = self.read_line()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File line (missing '+'): {s}")
            lines.append(s[1:])  # strip leading '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def find_context_core(
    lines: List[str], context: List[str], start: int
) -> Tuple[int, int]:
    """
    Find where a list of context lines appears in the file using fuzzy matching.

    This function implements a three-tier matching strategy:
    1. Try exact match first (returns fuzz=0)
    2. Try rstrip() match (ignoring trailing whitespace, returns fuzz=1)
    3. Try strip() match (ignoring all whitespace, returns fuzz=100)

    Args:
        lines: The file content as a list of lines
        context: The context lines to find
        start: The index to start searching from

    Returns:
        Tuple of (line_index, fuzz_score) where:
        - line_index: Index where context was found (-1 if not found)
        - fuzz_score: How fuzzy the match was (0=exact, 1=rstrip, 100=strip)
    """
    if not context:
        return start, 0

    # Try exact match first (best case)
    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0

    # Try rstrip match (ignoring trailing whitespace)
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [
            s.rstrip() for s in context
        ]:
            return i, 1

    # Try strip match (ignoring all leading/trailing whitespace)
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [
            s.strip() for s in context
        ]:
            return i, 100

    # Not found
    return -1, 0


def find_context(
    lines: List[str], context: List[str], start: int, eof: bool
) -> Tuple[int, int]:
    """
    Find context lines with special handling for end-of-file markers.

    When eof=True, the context is expected to be at the end of the file.
    We first try to find it there, and if that fails, we search from the
    start position but add a large fuzz penalty (10000) to indicate the
    EOF marker was in the wrong place.

    Args:
        lines: The file content as a list of lines
        context: The context lines to find
        start: The index to start searching from
        eof: Whether this context should be at end-of-file

    Returns:
        Tuple of (line_index, fuzz_score)
    """
    if eof:
        # For EOF context, try to find it at the end of the file first
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        # If not found at EOF, search from start but add large fuzz penalty
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(
    lines: List[str], index: int
) -> Tuple[List[str], List[Chunk], int, bool]:
    """
    Parse a section of an update patch to extract context lines and change chunks.

    This function reads lines marked with ' ' (context), '-' (delete), or '+' (insert)
    until it hits a section boundary (@@, ***, or end of input). It builds up:
    - Context lines (the "old" version including deleted lines)
    - Chunks of insertions/deletions

    Args:
        lines: The full patch text as a list of lines
        index: The starting index for this section

    Returns:
        Tuple of (context, chunks, end_index, is_eof) where:
        - context: List of context lines (what the file should look like before changes)
        - chunks: List of Chunk objects describing the changes
        - end_index: Index where this section ends
        - is_eof: True if this section ends with *** End of File marker
    """
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"Invalid Line: {s}")
        index += 1

        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(f"Invalid Line: {s}")
        s = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise DiffError("Nothing in this section")
    return old, chunks, index, False


# --------------------------------------------------------------------------- #
#  Patch → Commit and Commit application
# --------------------------------------------------------------------------- #
def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(
                f"{path}: chunk.orig_index {chunk.orig_index} exceeds file length"
            )
        if orig_index > chunk.orig_index:
            raise DiffError(
                f"{path}: overlapping chunks at {orig_index} > {chunk.orig_index}"
            )

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


def patch_to_commit(patch: Patch, orig: Dict[str, str]) -> Commit:
    commit = Commit()
    for path, action in patch.actions.items():
        if action.type is ActionType.DELETE:
            commit.changes[path] = FileChange(
                type=ActionType.DELETE, old_content=orig[path]
            )
        elif action.type is ActionType.ADD:
            if action.new_file is None:
                raise DiffError("ADD action without file content")
            commit.changes[path] = FileChange(
                type=ActionType.ADD, new_content=action.new_file
            )
        elif action.type is ActionType.UPDATE:
            new_content = _get_updated_file(orig[path], action, path)
            commit.changes[path] = FileChange(
                type=ActionType.UPDATE,
                old_content=orig[path],
                new_content=new_content,
                move_path=action.move_path,
            )
    return commit


# --------------------------------------------------------------------------- #
#  User-facing helpers
# --------------------------------------------------------------------------- #
def text_to_patch(text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
    lines = text.splitlines()  # preserves blank lines, no strip()
    if (
        len(lines) < 2
        or not Parser._norm(lines[0]).startswith("*** Begin Patch")
        or Parser._norm(lines[-1]) != "*** End Patch"
    ):
        raise DiffError("Invalid patch text - missing sentinels")

    parser = Parser(current_files=orig, lines=lines, index=1)
    parser.parse()
    return parser.patch, parser.fuzz


def identify_files_needed(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Update File: ") :]
        for line in lines
        if line.startswith("*** Update File: ")
    ] + [
        line[len("*** Delete File: ") :]
        for line in lines
        if line.startswith("*** Delete File: ")
    ]


def identify_files_added(text: str) -> List[str]:
    lines = text.splitlines()
    return [
        line[len("*** Add File: ") :]
        for line in lines
        if line.startswith("*** Add File: ")
    ]


# --------------------------------------------------------------------------- #
#  File-system helpers
# --------------------------------------------------------------------------- #
def load_files(paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
    return {path: open_fn(path) for path in paths}


def apply_commit(
    commit: Commit,
    write_fn: Callable[[str, str], None],
    remove_fn: Callable[[str], None],
) -> None:
    for path, change in commit.changes.items():
        if change.type is ActionType.DELETE:
            remove_fn(path)
        elif change.type is ActionType.ADD:
            if change.new_content is None:
                raise DiffError(f"ADD change for {path} has no content")
            write_fn(path, change.new_content)
        elif change.type is ActionType.UPDATE:
            if change.new_content is None:
                raise DiffError(f"UPDATE change for {path} has no new content")
            target = change.move_path or path
            write_fn(target, change.new_content)
            if change.move_path:
                remove_fn(path)


def open_file(path: str) -> str:
    with open(path, "rt", encoding="utf-8") as fh:
        return fh.read()


def write_file(path: str, content: str) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wt", encoding="utf-8") as fh:
        fh.write(content)


def remove_file(path: str) -> None:
    pathlib.Path(path).unlink(missing_ok=True)



def apply_patch(
    text: str,
    open_fn: Callable[[str], str] = open_file,
    write_fn: Callable[[str, str], None] = write_file,
    remove_fn: Callable[[str], None] = remove_file,
) -> str:
    """
    Main entry point: parse and apply a patch to the filesystem.

    This function orchestrates the entire patching process:
    1. Validates the patch has proper *** Begin Patch / *** End Patch markers
    2. Identifies which files need to be read (for UPDATE and DELETE operations)
    3. Loads those files from disk
    4. Parses the patch text into a structured Patch object
    5. Converts the Patch to a Commit with fully-resolved file contents
    6. Applies the commit to the filesystem

    Args:
        text: The patch text starting with "*** Begin Patch"
        open_fn: Function to read files (path -> content)
        write_fn: Function to write files (path, content -> None)
        remove_fn: Function to delete files (path -> None)

    Returns:
        "Done!" on success

    Raises:
        DiffError: If the patch is malformed or can't be applied

    Integration with Model:
    -----------------------
    The model learns to generate patches in the expected format during training.
    When the model wants to make code changes, it outputs a patch which is then
    passed to this function for execution. The custom open/write/remove functions
    allow sandboxing the file operations if needed.

    Example Patch:
    --------------
    *** Begin Patch
    *** Add File: new_file.py
    +def hello():
    +    print("Hello, world!")
    *** Update File: existing_file.py
    @@ import sys
     import sys
    +import os

    *** End Patch
    """
    if not text.startswith("*** Begin Patch"):
        raise DiffError("Patch text must start with *** Begin Patch")

    # Identify files that need to be read (UPDATE and DELETE require existing content)
    paths = identify_files_needed(text)

    # Load the current content of those files
    orig = load_files(paths, open_fn)

    # Parse the patch text into a Patch object with fuzzy matching
    patch, _fuzz = text_to_patch(text, orig)

    # Convert Patch to Commit (resolves all chunks into full file contents)
    commit = patch_to_commit(patch, orig)

    # Apply the commit to the filesystem
    apply_commit(commit, write_fn, remove_fn)

    return "Done!"


def main() -> None:
    import sys

    patch_text = sys.stdin.read()
    if not patch_text:
        print("Please pass patch text through stdin", file=sys.stderr)
        return
    try:
        result = apply_patch(patch_text)
    except DiffError as exc:
        print(exc, file=sys.stderr)
        return
    print(result)


if __name__ == "__main__":
    main()
