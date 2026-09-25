# Release workflow

Update an existing checkout without deleting the repository or its history. Validate the intended commit before creating a new version tag. Do not move an existing published tag.

## Validate

```bash
python scripts/update_manifest.py
python scripts/validate_release.py
PYTHONPATH=src pytest -q
python scripts/build_release.py
```

## Commit and tag

Commit the updated documentation, version metadata, and manifest together. After the commit's checks pass, create a new `v1.0.3` tag on that commit and publish the release. Preserve `v1.0.2` as the earlier analysis release.

Generated ZIP, tar.gz, and checksum files are written to `dist/`. Attach the complete release archives, not a partial-update ZIP. Confirm the published release contains the intended version before using its URL in the article.
