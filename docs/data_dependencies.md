# Data Dependencies

This document summarizes startup and optional external data dependencies for the current ROS-ABM runtime.

## 1. Required tracked startup data

- `new_populated.txt` is the default startup cell-location file used by `Model_initialParameters.py`.
- With default settings, it is resolved from the current working directory (`os.getcwd()`), so runs are expected from repository root unless an override is provided.

## 2. Removed/untracked HNSCC dependency

- `HNSCC_Broad_scRNAseq/` is not tracked in this repository.
- Historical/legacy workflows may still reference this directory name, but datasets must be supplied externally.

## 3. `load_gene_expression_data` expectations

`Simulation.load_gene_expression_data(metadata_path=None, expression_dir=None, cwd=None)` expects:

- `metadata_path`: CSV containing a `cell` column used to map per-cell expression files.
- `expression_dir`: directory containing per-cell CSV files named `<cell_id>.csv`.
- `cwd`: base directory used only for default path resolution when explicit paths are omitted.

## 4. Legacy defaults currently in code

If not overridden:

- `cwd` defaults to `os.getcwd()`.
- `expression_dir` defaults to `os.path.join(cwd, "HNSCC_Broad_scRNAseq")`.
- `metadata_path` defaults to a legacy absolute machine-specific OneDrive path in `Simulation.py`.

These defaults are preserved as legacy runtime contracts and should not be treated as portable dataset configuration.

## 5. Recommended external dataset usage

When using gene-expression loading paths:

- Always provide explicit `metadata_path` and `expression_dir` values that point to accessible local data.
- Keep external datasets outside this repository or in ignored paths.
- Treat gene-expression loading as optional/legacy unless explicitly invoked by your workflow.

