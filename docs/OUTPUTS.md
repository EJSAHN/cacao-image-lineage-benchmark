# Output schema

Generated filenames preserve immutable provenance identifiers used by the released checksums and Supplementary Data workbook. They do not denote alternative numerical methods.

The final analysis units are:

- **Exact component:** byte-identical file payloads.
- **Strict image lineage:** exact components connected by strong transformed-image evidence.
- **Extended image lineage:** strict lineages plus moderate transformed-image evidence.
- **Conservative split block:** image lineages linked by verified same-scene or acquisition-context evidence; all members must remain in the same evaluation partition.

Label eligibility is decided at the strict-lineage level. Same-scene edges constrain partitioning but do not force labels to agree.
