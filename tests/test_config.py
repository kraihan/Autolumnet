"""Smoke tests for CLI parsing and config merging."""
from autolumnet.utils.config import (parse_overrides, load_config, _set_nested)


def test_parse_overrides_dotted_keys():
    out = parse_overrides(["train.batch_size=1024", "loss.align=0.3", "model.rho=0.15"])
    assert out["train"]["batch_size"] == 1024
    assert out["loss"]["align"]       == 0.3
    assert out["model"]["rho"]        == 0.15


def test_parse_overrides_type_coercion():
    out = parse_overrides([
        "a.b=42", "a.c=3.14", "a.d=true", "a.e=False", "a.f=hello"
    ])
    assert out["a"]["b"] == 42 and isinstance(out["a"]["b"], int)
    assert out["a"]["c"] == 3.14 and isinstance(out["a"]["c"], float)
    assert out["a"]["d"] is True
    assert out["a"]["e"] is False
    assert out["a"]["f"] == "hello"


def test_load_config_with_overrides(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("train:\n  lr: 0.001\n  batch_size: 8\nmodel:\n  rho: 0.20\n")
    cfg = load_config(cfg_file, overrides={"train": {"batch_size": 1024}})
    assert cfg["train"]["lr"] == 0.001          # untouched
    assert cfg["train"]["batch_size"] == 1024   # overridden
    assert cfg["model"]["rho"] == 0.20          # other branches preserved


def test_dotdict_attribute_access(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("train:\n  lr: 0.001\n")
    cfg = load_config(cfg_file)
    assert cfg.train.lr == 0.001
