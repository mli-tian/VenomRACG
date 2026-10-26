"""Backward-compatible re-exports of the shared dual-language utilities.

The implementation lives in ``data_process/attack_util.py``; this shim keeps
``from utils import ...`` working for scripts inside ``importance_metadata/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack_util import (  # noqa: F401
    code_to_tokenized,
    get_identifiers,
    get_parser,
    remove_comments_and_docstrings,
)
