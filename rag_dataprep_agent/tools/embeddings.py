from __future__ import annotations

import hashlib
import math


def deterministic_embedding(text: str, dimensions: int = 32) -> list[float]:
    """Small local embedding substitute for tests and no-key demo runs."""
    values = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = digest[0] % dimensions
        sign = 1.0 if digest[1] % 2 == 0 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 6) for value in values]
