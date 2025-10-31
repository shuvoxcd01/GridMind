# Scripts Directory

This directory contains utility scripts for the GridMind project.

## update_citation_version.py

This script automatically updates the version in `CITATION.cff` to match the version specified in `pyproject.toml`.

### Usage

```bash
python scripts/update_citation_version.py
```

### Integration with Git Hooks

This script is automatically run by the pre-commit hook whenever `pyproject.toml` or `CITATION.cff` are modified. The hook ensures that the version in the citation file stays synchronized with the project version.

### Pre-commit Hook Setup

The repository includes:
1. A Git pre-commit hook (`.git/hooks/pre-commit`)
2. A pre-commit configuration file (`.pre-commit-config.yaml`) for use with the `pre-commit` tool

#### Using with pre-commit tool (Optional)

If you want to use the `pre-commit` tool instead of the Git hook:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the hooks:
   ```bash
   pre-commit install
   ```

3. Run manually (optional):
   ```bash
   pre-commit run --all-files
   ```

### How it Works

1. When you commit changes to `pyproject.toml`, the hook detects the change
2. The script extracts the version from `pyproject.toml`
3. It updates the version field in `CITATION.cff`
4. If changes were made, the updated `CITATION.cff` is automatically staged for commit
