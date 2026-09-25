# Code provenance

This public release was assembled from the final executed workflow sources. Obsolete packages and failed implementations were excluded. Three production corrections are incorporated directly:

1. RAR extraction prefers a modern user-specified or environment-provided `7zz` executable and reports extractor output on failure.
2. Lineage-weighted model arrays are copied before normalization so they remain writable with pandas copy-on-write behavior.
3. Visual-feature extraction imports scikit-learn metrics lazily because the feature environment requires PyTorch but not scikit-learn.

Repair scripts, job logs, private prompts, collaborator correspondence and publication-figure generation code are not part of this repository.
