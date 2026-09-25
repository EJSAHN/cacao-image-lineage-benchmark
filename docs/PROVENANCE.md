# Data and analysis provenance

Source imagery is obtained from the Figshare and Zenodo records listed in `config/datasets.json`. Download metadata, file checksums, source-relative paths, and original labels are retained by the analysis workflow.

The reconstruction distinguishes byte-identical file groups, transformed-photograph lineages, and acquisition-scene blocks. Source identifiers and reconstructed group identifiers connect the photograph manifest to the benchmark tables in `data/derived/`.

`docs/PIPELINE.md` describes the analysis sequence, and `docs/OUTPUTS.md` defines the grouping levels. The implemented procedures are in `src/cacao_image_benchmark/workflow/`.

Raw photographs remain subject to their source licenses. Publication plotting scripts and their input tables are provided as Supplementary Code S1.
