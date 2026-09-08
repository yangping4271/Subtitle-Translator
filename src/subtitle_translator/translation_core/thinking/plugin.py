"""思考插件的注册与加载。"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Iterable, Optional

from .types import ThinkingPlugin

_EXTRA_PLUGINS: list[ThinkingPlugin] = []
_BUILTIN_PLUGINS: Optional[tuple[ThinkingPlugin, ...]] = None


def register_plugin(plugin: ThinkingPlugin) -> None:
    """运行时追加一条适配规则；测试和外部扩展用。"""
    _EXTRA_PLUGINS.append(plugin)
    clear_plugin_cache()


def unregister_plugin(plugin: ThinkingPlugin) -> None:
    """移除 register_plugin 追加的规则。"""
    _EXTRA_PLUGINS.remove(plugin)
    clear_plugin_cache()


def clear_plugin_cache() -> None:
    global _BUILTIN_PLUGINS
    _BUILTIN_PLUGINS = None


def _load_builtin_plugins() -> tuple[ThinkingPlugin, ...]:
    from . import plugins as plugins_pkg

    loaded: list[ThinkingPlugin] = []
    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(
            f"{plugins_pkg.__name__}.{module_info.name}"
        )
        loaded.extend(getattr(module, "PLUGINS", ()))
    return tuple(loaded)


def iter_plugins() -> Iterable[ThinkingPlugin]:
    global _BUILTIN_PLUGINS
    if _BUILTIN_PLUGINS is None:
        _BUILTIN_PLUGINS = _load_builtin_plugins()
    yield from _BUILTIN_PLUGINS
    yield from _EXTRA_PLUGINS
