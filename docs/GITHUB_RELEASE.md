# Source distributions

From a clean checkout with the package and test dependencies installed, run:

```bash
python scripts/validate_release.py
PYTHONPATH=src pytest -q
python scripts/build_release.py
```

The build creates a ZIP archive, a tar.gz archive, and a SHA-256 checksum file in `dist/`. Archive names use the package name and version in `pyproject.toml`.

For a new software version, keep `pyproject.toml`, `CITATION.cff`, and `src/cacao_image_benchmark/__init__.py` consistent. After editing tracked files, regenerate `MANIFEST.sha256.tsv` with `python scripts/update_manifest.py`, then repeat validation and testing. Review the changes and commit the files and manifest together.

Create the version tag from the validated commit. Distribute the complete source archive with its checksums, and cite that tag when referring to a specific version.
