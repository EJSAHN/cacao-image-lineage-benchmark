# GitHub release workflow

Keep the repository private until the manuscript team and institutional release process approve public access. The commands below assume Git and the GitHub CLI are installed.

## First push

```bash
git init
git branch -M main
git add .
git commit -m "Initial analysis release"
gh auth login
gh repo create cacao-image-lineage-benchmark --private --source=. --remote=origin --push
```

## Validate before a release

```bash
python scripts/validate_release.py
pytest
python scripts/build_release.py
```

## Create the version tag

```bash
git add .
git commit -m "Prepare v1.0.1"
git tag -a v1.0.1 -m "Cacao image lineage benchmark v1.0.1"
git push origin main --tags
```

The generated ZIP, tar.gz and checksum file are written to `dist/`. Attach them to the GitHub release after public-release approval.
