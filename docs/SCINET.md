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

## Working storage

`/90daydata` is temporary and is not backed up. The helper below updates file timestamps in the working cache and writes an audit report; it does not provide a backup:

```bash
bash scripts/refresh_working_data.sh "$CACAO_WORKING_ROOT"
```

Keep code, derived tables and release archives in persistent project storage. See the SCINet storage policy: https://scinet.usda.gov/guides/data/storage
