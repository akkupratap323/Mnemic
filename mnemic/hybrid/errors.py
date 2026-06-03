"""
Copyright 2026, Abishek (Mnemic project).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

from __future__ import annotations


class HybridMemoryError(Exception):
    """Base class for all hybrid-memory errors."""


class InvalidInput(HybridMemoryError, ValueError):
    """Raised when caller-supplied input fails validation."""


class DimensionMismatch(HybridMemoryError, ValueError):
    """Raised when an embedding's dimension does not match the store."""


class DuplicateItem(HybridMemoryError, ValueError):
    """Raised when adding an item whose id already exists."""
