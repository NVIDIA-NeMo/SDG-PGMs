# US Person Generator Example

This example mirrors the internal US person PGM structure while using tiny dummy data assets that can be published with the repo.

The generator accepts either:

- `data_path`, a tar file with the expected parquet members listed in `PERSON_DATA_TARFILE_PATHS`
- `data`, an in-memory `USPersonData` instance

The checked-in dummy assets are:

- `examples/us_person/data/person_data_v4.tar`
- `examples/us_person/data/zip_area_code_map.parquet`

They preserve the real tar member names and categorical structure, but use synthetic ZIP codes, places, names, streets, and perturbed counts. Regenerate them with:

```bash
uv run python -m examples.us_person.build_dummy_data
```

For local validation against internal assets, set `US_PERSON_DATA_PATH` and `US_PERSON_AREA_CODE_DATA_PATH` before constructing `USPersonGenerator`.
