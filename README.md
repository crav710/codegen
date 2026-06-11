# codegen — AIML PGCP Capstone (CodeGen / Task 3)

**Repo:** https://github.com/crav710/codegen  
(Team fork of https://github.com/svamsikrishna54/codgen)

## Project layout

| File | Purpose |
|------|---------|
| `week1_setup_and_baselines.ipynb` | **Single source of truth** — setup, baselines, mentor extension (all code inline) |
| `requirements.txt` | Reference package list (Colab installs these in notebook cell 1.3) |
| `results/baselines.csv` | Saved baseline run results |

There is no `lib/` package — edit the notebook cells directly (Part 1 cells 1.5–1.11).

## Colab quick start

1. Open `week1_setup_and_baselines.ipynb` in Google Colab (upload or open from GitHub)
2. Runtime → **T4 GPU**
3. Run **Part 1** — setup + inline library cells
4. Run **Part 2** for K=0 baselines
5. Run **Part 3** for mentor dataset extension

## Mentor task (Part 3)

Extends each benchmark row to **NL + PL1 (Python) + PL2 (Java)**:

- **HumanEval / MBPP:** generate PL2 from NL+PL1, validate, flag `pl2_valid`
- **HumanEval-X:** generate NL from PL1+PL2, validate via regeneration, flag `nl_valid`

Output: `extended/unified_dataset.jsonl` on Google Drive under `CODEGEN_DATA_DIR` (default: `codegen_week1/`)
