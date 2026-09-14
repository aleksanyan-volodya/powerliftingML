# Powerlifting ML - Regression Challenge

A Kaggle competition repository for a regression problem predicting powerlifting performance metrics.

## Project Structure

```
.
├── data/
│   ├── raw/              # Original Kaggle data (not committed)
│   ├── processed/        # Cleaned/processed datasets
│   └── external/         # Additional external data sources
├── notebooks/            # Jupyter notebooks for exploration
├── src/
│   ├── features/         # Feature engineering
│   ├── models/           # Model definitions & training
│   └── utils/            # Utility functions
├── tests/                # Unit tests
├── submissions/          # Kaggle submission files
├── pyproject.toml        # Project metadata & dependencies
└── README.md
```

## Setup

### Prerequisites
- Python 3.13+
- Git

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd powerliftingML
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

Or with `uv` (faster):
```bash
uv sync
```

## Collaboration

- Use feature branches for new work: `git checkout -b feature/description`
- Keep commits focused and descriptive
- Write tests for new features
- Ensure code passes linting before merging

## Data

Download Kaggle data to `data/raw/`:
```bash
kaggle competitions download -c <competition-name> -p data/raw/
```

## Development

- Exploratory work in `notebooks/`
- Production code in `src/`
- Add dependencies in `pyproject.toml`

## Submission

Save predictions to `submissions/` following Kaggle's format (e.g., `submission_v1.csv`).
