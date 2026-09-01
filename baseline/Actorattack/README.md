# Actorattack preattack

This folder contains the files needed to run the `preattack` workflow from the original `ActorAttack-master` project.

## Layout
- `preattack.py`, `config.py`, `utils.py`: core code and configuration.
- `prompts/`: prompt templates used by the pipeline.
- `data/`: default harmful-goal CSVs (defaults to `data/harmbench.csv`).
- `requirements.txt`: minimal dependencies for preattack.

## Setup
1. Install dependencies (Python 3.11+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Provide API credentials via environment variables (only the ones you use are required):
   - `GPT_API_KEY`, `BASE_URL_GPT`
   - `CLAUDE_API_KEY`, `BASE_URL_CLAUDE`
   - `DEEPSEEK_API_KEY`, `BASE_URL_DEEPSEEK`
   - `DEEPINFRA_API_KEY`, `BASE_URL_DEEPINFRA`

## Run
From this folder:
```bash
python preattack.py
```
Key runtime options live in `PreAttackConfig` inside `config.py` (model name, actor count, CSV path, prompt paths). Results are written to `./pre_attack_result/`.
