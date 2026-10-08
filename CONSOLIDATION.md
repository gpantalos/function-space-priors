# Consolidated projects

Continue development of these modules in this repository. Each retains its original source tree and dependency files; the launcher sets its working directory and import path and uses an isolated Python environment.

| Source | Maintained module | Original commit |
|---|---|---|
| `gpantalos/thesis` | [research/thesis](research/thesis) | `19a7c80858cb` |

## Running tools

Use `uv run --no-project workspace.py --list` to see tasks and `uv run --no-project workspace.py --dry-run TASK` to inspect a command. Append arguments after the task name.

| Command | Feature |
|---|---|

Original module README files describe external prerequisites and data. The thesis module retains its PACOH/FVI scripts and environments; follow `research/thesis/pacoh/README.md` for its original research setup.

## History

Original default-branch commits are ancestors of the import merge commits. Other remote branches and tags are retained under `consolidated/SOURCE/`. Use `git log --all` to inspect original paths. `CONSOLIDATION.json` records commits, source tree IDs and module paths. Before migration, all source and existing destination refs were saved in verified Git bundles. Local-only work remains preserved separately and is not silently mixed into maintained code.

## Runtime scope

The consolidation keeps distinct algorithms, brokers and services as separately invoked modules. It does not combine them into a single training loop, trading engine or web app. External credentials, datasets, weights, native libraries and schedules still require setup. Flight notification delivery and stock messaging were unfinished in the source projects. No live broker, outbound notification, paid generation or training job was started during the migration.

## Thesis methods

The research/thesis directory retains the PACOH, FVI, functional-BNN and gradient-estimator implementations with their original imports and environment requirements. This is a complete research-source consolidation; those methods were not rewritten as fpbnn CLI commands or evaluated against the current TensorFlow environment.
