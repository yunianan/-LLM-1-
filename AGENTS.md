# Repository instructions

When adding, removing, renaming, or changing an L1-L5 Pydantic model or TypedDict,
run `python tools/generate_schema_reference.py --write` and commit the updated
`docs/layer-schema-reference.md`. The schema-reference consistency test must pass.
