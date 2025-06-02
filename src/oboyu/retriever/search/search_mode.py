"""Search mode enumeration."""

from enum import Enum


class SearchMode(Enum):
    """Available search modes."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
