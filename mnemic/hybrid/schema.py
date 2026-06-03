"""
Copyright 2026, Abishek (Mnemic project).

Code-aware entity and edge types for the graph tier. Pass CODE_ENTITY_TYPES /
CODE_EDGE_TYPES to Mnemic.remember(entity_types=..., edge_types=...) to make the
knowledge graph understand repos, files, decisions, bugs, and their relations.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Repo(BaseModel):
    """A source-code repository."""

    url: str | None = Field(default=None, description='Clone or web URL')
    default_branch: str | None = Field(default=None, description='e.g. main')


class File(BaseModel):
    """A file within a repository."""

    path: str | None = Field(default=None, description='Repo-relative path')
    language: str | None = Field(default=None, description='Programming language')


class Function(BaseModel):
    """A function, method, or class."""

    name: str | None = None
    signature: str | None = None
    file_path: str | None = None


class Decision(BaseModel):
    """An engineering decision worth remembering."""

    summary: str | None = None
    rationale: str | None = None


class Bug(BaseModel):
    """A defect and its status."""

    description: str | None = None
    severity: str | None = Field(default=None, description='low | medium | high | critical')
    fixed: bool = False


class Task(BaseModel):
    """A unit of work."""

    title: str | None = None
    status: str | None = Field(default=None, description='todo | in_progress | done')


class Dependency(BaseModel):
    """A package or library dependency."""

    name: str | None = None
    version: str | None = None


class ApiEndpoint(BaseModel):
    """An HTTP API endpoint."""

    method: str | None = None
    path: str | None = None


class DependsOn(BaseModel):
    """Source depends on target."""


class Implements(BaseModel):
    """Source implements target (e.g. a function implements a requirement)."""


class FixedBy(BaseModel):
    """A bug fixed by a change."""

    commit: str | None = None


class Supersedes(BaseModel):
    """A decision/fact that replaces an earlier one."""


CODE_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Repo': Repo,
    'File': File,
    'Function': Function,
    'Decision': Decision,
    'Bug': Bug,
    'Task': Task,
    'Dependency': Dependency,
    'ApiEndpoint': ApiEndpoint,
}

CODE_EDGE_TYPES: dict[str, type[BaseModel]] = {
    'DependsOn': DependsOn,
    'Implements': Implements,
    'FixedBy': FixedBy,
    'Supersedes': Supersedes,
}
