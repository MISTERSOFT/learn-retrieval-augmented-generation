# learn-retrieval-augmented-generation

Course from https://www.boot.dev/

# Prerequisites

1. Install the Python package and project manager [UV](https://docs.astral.sh/uv/getting-started/installation/).

2. Download the data by running the following command:
```bash
scripts/download-data.sh
```

# Commands
Run CLI:
```bash
uv run .\cli\keyword_search_cli.py search "furious fast"
```

Run linter:
```bash
uv run ruff check
```

Run formatter:
```bash
uv run ruff format
```