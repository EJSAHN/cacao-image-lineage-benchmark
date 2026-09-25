# Cacao Image Lineage Benchmark

This repository reconstructs file identity, transformed-image lineage and acquisition-scene dependence across public cacao disease image collections. It then compares conventional image-wise evaluation with exact-component, strict-lineage and scene-blocked validation.

The repository provides Python workflows, configuration files, derived tables, and regression tests. Source photographs are obtained from the original repositories. Publication plotting scripts and their input tables are provided as Supplementary Code S1.

## Main workflow

1. Download the Figshare and Zenodo records listed in `config/datasets.json`.
2. Inventory and extract the archives while preserving source provenance.
3. Build the canonical photograph manifest and harmonized label ontology.
4. Detect exact copies, transformed descendants and same-scene captures.
5. Freeze strict image lineages and conservative split blocks.
6. Compare naive, lineage-safe and scene-safe classification designs.
7. Run matched-removal, shortcut and end-to-end sensitivity analyses.

## Installation

```bash
conda env create -f environment.yml
conda activate cacao-image-lineage-benchmark
pip install -e .
cacao-benchmark validate-install
```

A pip-only installation is also possible:

```bash
python -m venv ../cacao-benchmark-env
source ../cacao-benchmark-env/bin/activate
pip install -e '.[full]'
```

A recent `7zz` executable is required to extract the RAR archives. The conda environment installs it.

## Command-line interface

```bash
cacao-benchmark list
cacao-benchmark step-help download_public_data
```

Each command forwards its remaining options to the corresponding workflow module. For example:

```bash
source config/paths.env.example
cacao-benchmark run download_public_data   --root "$CACAO_PROJECT_ROOT"   --big "$CACAO_WORKING_ROOT"   --figshare-id 31294003   --zenodo-id 17716661
```

The complete command order and expected inputs are described in `docs/PIPELINE.md`. SCINet execution examples are in `docs/SCINET.md`.

## Released derived data

`data/derived/` contains compact tables, `Supplementary_Data_S1.xlsx`, and the complete photograph manifest in `Supplementary_Data_S1_manifest.tsv.gz`. These files store archive counts, lineage and split audits, verification summaries and the principal benchmark results. The workbook points to the separate manifest; distribute both together. They contain no raw photographs.

## Testing

```bash
python scripts/validate_release.py
PYTHONPATH=src pytest -q
```

The tests cover selected numerical and partitioning behavior on small fixtures. They do not rerun the complete image analysis. Publication plotting code is provided as Supplementary Code S1.

## Reproducibility boundary

Fixed seeds are specified for the recorded evaluation designs; bitwise agreement across software versions and hardware is not guaranteed. Image downloads remain governed by the source repositories and their licenses. Dataset owners may update records after this release; the downloader stores API metadata and file checksums so that retrieved payloads can be audited.

## Citation

Use `CITATION.cff` to cite the software.

## License and source-data notice

Code and original repository metadata are released under the terms in `LICENSE`. Source images are not relicensed by this repository; consult the original records and archive-specific terms described in `NOTICE.md`.
