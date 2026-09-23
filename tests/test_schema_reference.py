from tools.generate_schema_reference import OUTPUT, render


def test_layer_schema_reference_matches_code():
    assert OUTPUT.read_text(encoding="utf-8") == render(), (
        "Schema 已变化但文档未同步；请运行 "
        "python tools/generate_schema_reference.py --write"
    )
