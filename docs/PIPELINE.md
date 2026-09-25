# Reproduction pipeline

| Order | Command | Purpose |
|---:|---|---|
| 1 | `download_public_data` | Download and checksum the public records. |
| 2 | `inventory_archives` | Inventory archive members before extraction. |
| 3 | `build_archive_manifest` | Create one extraction task per source archive. |
| 4 | `extract_archive` | Extract one archive; use a job array for multiple archives. |
| 5 | `build_canonical_manifest` | Classify photographs, masks and annotations and calculate exact file and decoded-pixel hashes. |
| 6 | `harmonize_labels` | Apply the harmonized pathology and stage ontology and reconcile public split files. |
| 7 | `compute_perceptual_hashes` | Calculate orientation-aware perceptual hashes for exact-component representatives. |
| 8 | `discover_perceptual_candidates` | Generate transformed-lineage candidates and metadata-only controls. |
| 9 | `prepare_geometric_verification` | Create candidate and control pair tables. |
| 10 | `verify_image_pairs` | Apply the SIFT/ORB/RANSAC and photometric verifier. |
| 11 | `build_initial_lineages` | Aggregate controls and construct initial strict and scene-aware families. |
| 12 | `extract_image_embeddings` | Extract frozen ResNet-18 and EfficientNet-B0 embeddings. |
| 13 | `retrieve_embedding_candidates` | Run the exact blockwise cosine-neighbor sweep. |
| 14 | `rank_embedding_candidates` | Apply the calibrated evidence-first candidate gate. |
| 15 | `prepare_primary_verification` | Prepare the 6,000-pair primary geometric tranche. |
| 16 | `verify_primary_pairs` | Verify one primary pair-table shard. |
| 17 | `aggregate_primary_verification` | Aggregate primary results and estimate residual discovery yield. |
| 18 | `prepare_reserve_verification` | Prepare the remaining evidence-gated reserve. |
| 19 | `verify_reserve_pairs` | Verify one reserve pair-table shard. |
| 20 | `freeze_lineages_and_split_blocks` | Freeze strict image lineages, capture bursts, conservative split blocks and task-ready units. |
| 21 | `prepare_frozen_benchmark` | Build repeated naive, exact, lineage, scene and source-holdout assignments. |
| 22 | `run_frozen_benchmark` | Run one frozen-feature benchmark configuration. |
| 23 | `aggregate_frozen_benchmark` | Aggregate tabular frozen-feature benchmark results. |
| 24 | `prepare_design_confirmation` | Build repeated-seed, fixed-test, training-weight and source-holdout confirmation designs. |
| 25 | `run_design_confirmation` | Run one design-confirmation configuration. |
| 26 | `aggregate_design_confirmation` | Aggregate repeated and paired confirmation results. |
| 27 | `prepare_matched_and_shortcut_controls` | Prepare matched-removal nulls and shortcut-control designs. |
| 28 | `extract_shortcut_features` | Extract one visual-shortcut representation. |
| 29 | `run_matched_removal` | Run one matched clean-removal replicate. |
| 30 | `run_shortcut_benchmark` | Run one shortcut-control benchmark. |
| 31 | `aggregate_control_analyses` | Aggregate matched-removal and shortcut-control tables. |
| 32 | `prepare_end_to_end_sensitivity` | Prepare repeated path-random and scene-blocked CNN assignments. |
| 33 | `cache_training_images` | Build one shard of the standardized image cache. |
| 34 | `train_end_to_end_sensitivity` | Train one five-fold end-to-end configuration. |
| 35 | `aggregate_end_to_end_sensitivity` | Aggregate end-to-end metrics and paired deltas. |

Exact arguments for each command are available with:

```bash
cacao-benchmark step-help COMMAND
```

Large pair verification, model configuration and image-cache commands are designed for Slurm arrays. The table-producing aggregation commands do not generate manuscript figures; publication plotting scripts are provided as Supplementary Code S1.
