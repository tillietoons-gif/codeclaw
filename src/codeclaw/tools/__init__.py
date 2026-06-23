"""Concrete tool implementations.

Adding a tool: subclass `Tool`, give it a name/description/parameters schema,
implement `async run`, register it in `tools/__init__.py`.
"""
from .base import (
    ApprovalDecision,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    parse_args,
)
from .filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from .git import GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from .search import GrepTool
from .shell import ExecTool


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in (
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListDirTool(),
        GrepTool(),
        ExecTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
        GitCommitTool(),
    ):
        reg.register(t)
    return reg


__all__ = [
    "ApprovalDecision",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "parse_args",
]
