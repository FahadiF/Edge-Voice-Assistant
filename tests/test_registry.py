from __future__ import annotations

import pytest

from eva.core.errors import RegistryError
from eva.core.registry import Registry


def test_register_and_get() -> None:
    reg: Registry[int] = Registry("numbers")
    reg.register("one", 1)
    assert reg.get("one") == 1
    assert "one" in reg
    assert len(reg) == 1
    assert reg.ids() == ["one"]


def test_duplicate_rejected_unless_replace() -> None:
    reg: Registry[int] = Registry("numbers")
    reg.register("x", 1)
    with pytest.raises(RegistryError):
        reg.register("x", 2)
    reg.register("x", 2, replace=True)
    assert reg.get("x") == 2


def test_unknown_id_lists_known() -> None:
    reg: Registry[int] = Registry("numbers")
    reg.register("a", 1)
    with pytest.raises(RegistryError, match=r"unknown id 'b'.*a"):
        reg.get("b")


def test_unregister() -> None:
    reg: Registry[int] = Registry("numbers")
    reg.register("a", 1)
    reg.unregister("a")
    assert "a" not in reg
    with pytest.raises(RegistryError):
        reg.unregister("a")


def test_empty_id_rejected() -> None:
    reg: Registry[int] = Registry("numbers")
    with pytest.raises(RegistryError):
        reg.register("", 1)


def test_snapshot_is_a_copy() -> None:
    reg: Registry[int] = Registry("numbers")
    reg.register("a", 1)
    snap = reg.snapshot()
    snap["b"] = 2
    assert "b" not in reg
