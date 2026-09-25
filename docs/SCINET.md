# SCINet execution

Set project-specific paths and the Slurm account in the shell that submits jobs:

```bash
export CACAO_PROJECT_ROOT=/project/<account>/<user>/cacao-image-lineage-benchmark
export CACAO_WORKING_ROOT=/90daydata/<account>/<user>/cacao-image-lineage-benchmark-cache
export SLURM_ACCOUNT=<account>
```

Create the environment from a compute node rather than the login node:

```bash
module load miniconda
conda env create -p "$CACAO_PROJECT_ROOT/env" -f environment.yml
```

A generic array invocation is:

```bash
sbatch --account="$SLURM_ACCOUNT" --partition=ceres --array=1-N%CONCURRENCY   --wrap='source "$CACAO_PROJECT_ROOT/env/bin/activate"; cacao-benchmark run COMMAND ...'
```

Use `cacao-benchmark step-help COMMAND` to inspect required options before constructing an array job. Keep persistent tables and code under `/project`; use `/90daydata` only for downloadable archives, extracted images, embeddings and model caches.

## Refreshing a working cache

SCINet working storage is temporary. The helper below updates access and modification timestamps and writes an audit report:

```bash
bash scripts/refresh_working_data.sh "$CACAO_WORKING_ROOT"
```

This is not a backup. The code, compact derived tables and release archives should also be stored in persistent project storage and an external repository.

## Retention is not a backup

A timestamp-refresh helper does not provide guaranteed retention or a backup. SCINet states that cluster storage is not backed up, that files older than 90 days in short-term storage are deleted permanently, and that irreplaceable results should be transferred to appropriate long-term storage such as Juno. Consult the current policy rather than treating a successful `touch` or Slurm status as a preservation guarantee. Avoid redundant archival storage of publicly downloadable raw data.

Policy: https://scinet.usda.gov/guides/data/storage
