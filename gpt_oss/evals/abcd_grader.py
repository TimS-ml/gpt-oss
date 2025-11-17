"""
Multiple-Choice Answer Extraction Module

This module extracts multiple-choice answers (A, B, C, or D) from model-generated
text. It's used by evaluations like GPQA that present questions with four options.

The Challenge:
Models can express their answer in many different formats:
- "The answer is A"
- "**Answer:** (B)"
- "I choose option C"
- "\\boxed{D}"
- "**A) Description of option**"
- Just "A" on its own line

This module uses a prioritized list of regex patterns to handle all common formats,
from most specific (e.g., "**Answer:** A") to most general (e.g., bare "A").

Pattern Priority:
Patterns are tried in order, with more specific patterns first to avoid false matches.
For example, we match "Answer: A" before matching "(A)" to avoid extracting A from
"(Answer)" in unrelated text.

Usage:
    from .abcd_grader import extract_abcd

    response = "After analyzing the question, I believe **Answer: B** is correct."
    answer = extract_abcd(response)  # Returns "B"
"""

import re
import sys


# List of regex patterns for extracting A/B/C/D answers, ordered by priority
# More specific patterns come first to avoid false positives
_PATTERNS = [
    # 0)"**Answer:** A" or "*Answers* – B", i.e. markdown‐wrapped "Answer(s)" with an unwrapped letter.
    re.compile(
        r'''(?ix)                   # case‐insensitive, ignore‐space
        (?:\*{1,2}|_{1,2})          # leading *…*  or _…_
        Answer[s]?                  #   Answer or Answers
        \s*[:\-–]?                  #   optional separator
        (?:\*{1,2}|_{1,2})          # closing wrapper
        \s*                         # optional space
        ([ABCD])\b                  # the actual letter
        ''',
        re.X
    ),

    # 0.1)
    re.compile(r'''(?ix)           # ignore case, allow verbose mode
        ^\s*                      # optional leading whitespace
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper
        Answer:?                   # the word 'answer' with an optional colon
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper again
        \s*:?\s*                  # optional colon with optional spaces
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper before letter
        ([ABCD])                 # capture the letter
        (?:\*{1,2}|_{1,2})?       # optional markdown wrapper after letter
        \s*                     # optional trailing whitespace, end of line
    ''', re.MULTILINE),

    # 1) Answer: (C)   or   Answers: (B)
    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)'),

    # 2) Answer: C    or   Answers – D
    re.compile(r'(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b'),

    # 3) Option B   or   Choice: C
    re.compile(r'(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b'),

    # 7) LaTeX \boxed{...A...}, catches both \boxed{A} and
    #    \boxed{\text{A } 2.08\times10^{-6}\,\mathrm{m}} etc.
    re.compile(r'(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}', re.MULTILINE),

    # 7.5) LaTeX \boxed{\textbf{...C...}}
    re.compile(r'(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),

    # 7.51) LaTeX \boxed{\text{...C...}}
    re.compile(r'(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}', re.MULTILINE),

    # 4) bare singletons:  (A)  [B]
    re.compile(r'(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])'),

    # 5) Markdown‐wrapped: *A*  **B**  _C_  __D__
    re.compile(r'(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])'),

    # 6) LaTeX \textbf{...C...}
    re.compile(r'(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}'),

    # 8) markdown‐wrapped answer plus “)” plus description, e.g. **D) …**
    re.compile(r'''(?x)                        # ignore whitespace in pattern
        (?<![A-Za-z0-9])            # not preceded by word‐char
        (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
        \s*([ABCD])\)               # capture letter plus “)”
        [^*_\n]+?                   # some text inside wrapper
        (?:\*{1,2}|_{1,2})          # closing wrapper
        (?![A-Za-z0-9])             # not followed by word‐char
    '''),

    # 9) final fallback: a line that's exactly "A", "B.", "C)", "**D**", etc.
    re.compile(r'''(?x)^\s*
        (?:\*{1,2}|_{1,2})?     # optional markdown wrapper
        ([ABCD])                # capture group for letter
        (?:\*{1,2}|_{1,2})?     # optional closing markdown
        \s*[\.\)\-–:]?          # optional separator after the letter
        \s*.*$                  # allow any following text
    ''', re.MULTILINE),
]


def extract_abcd(text: str) -> str | None:
    """
    Extract a multiple-choice answer (A, B, C, or D) from model-generated text.

    This function tries all patterns in _PATTERNS in priority order, looking for
    the most specific match first. If multiple patterns match, it chooses based on:
    1. Pattern priority (earlier patterns are more specific)
    2. Match length (longer matches are more likely to be intentional)

    Args:
        text: Model-generated response text, may include:
            - Markdown formatting (**bold**, _italic_)
            - LaTeX formatting (\\boxed{}, \\textbf{})
            - Plain text answers
            - Mixed content with reasoning and answer

    Returns:
        Single character 'A', 'B', 'C', or 'D' if an answer is found
        Returns the first character of text (fallback) if no pattern matches
        This fallback handles cases like "A. Description..." where the answer
        is at the very start.

    Examples:
        >>> extract_abcd("The answer is **B**")
        'B'
        >>> extract_abcd("I believe (C) is correct")
        'C'
        >>> extract_abcd("\\\\boxed{A}")
        'A'
        >>> extract_abcd("**D) This option is best**")
        'D'
    """
    matches = []
    # Try all patterns and collect matches
    for prio, pat in enumerate(_PATTERNS):
        m = pat.search(text)
        if m:
            # Extract the captured letter (group 1) and normalize to uppercase
            letter = m.group(1).upper()
            if letter in 'ABCD':
                # Store priority, match object, and extracted letter
                matches.append((prio, m, letter))

    # Sort matches by:
    # 1. Priority (lower is better - earlier patterns are more specific)
    # 2. Match length (longer matches are more likely intentional)
    # This ensures we prefer "Answer: A" over just "(A)" if both exist
    matches.sort(key=lambda triple: (
        triple[0],  # Pattern priority
        len(triple[1].group(0))  # Length of matched text (negative for descending)
    ))

    # Return the best match if any patterns succeeded
    for _, match, letter in matches:
        return letter

    # Fallback: If no patterns match, try the first character after removing markdown
    # This handles cases like "**A**" at the very start of the response
    return text.removeprefix('**')[:1]


def main():
    """
    Command-line interface for testing answer extraction.

    Usage:
        # Test on files:
        python -m gpt_oss.evals.abcd_grader file1.txt file2.txt

        # Test on stdin:
        echo "The answer is B" | python -m gpt_oss.evals.abcd_grader

    This is useful for debugging answer extraction on actual model outputs.
    """
    if len(sys.argv) > 1:
        # Process files
        for fn in sys.argv[1:]:
            with open(fn, encoding='utf8') as fp:
                text = fp.read()
            ans = extract_abcd(text)
            print(f"{fn} ➜ {ans!r}")
    else:
        # Read from stdin
        for line in sys.stdin:
            ans = extract_abcd(line)
            print(f"{line} ➜ {ans!r}")


if __name__ == "__main__":
    main()

