# codgen — AIML PGCP Capstone (CodeGen / Task 3)

GitHub: https://github.com/svamsikrishna54/codgen

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Week 1 K=0 baselines only |
| `mentor-extension` | Baselines + mentor dataset extension (`run_extension`) |

## Colab quick start

1. Open `week1_setup_and_baselines.ipynb` in Google Colab
2. Runtime → **T4 GPU**
3. Run **Part 1** (cell 1.1 clones/pulls branch `mentor-extension`)
4. Run **Part 2** for baselines (optional if already done)
5. Run **Part 3** for mentor dataset extension

Override branch:
```python
import os
os.environ['CODEGEN_GIT_BRANCH'] = 'mentor-extension'
```

## Mentor task (Part 3)

Extends each benchmark row to **NL + PL1 (Python) + PL2 (Java)**:

- **HumanEval / MBPP:** generate PL2 from NL+PL1, validate, flag `pl2_valid`
- **HumanEval-X:** generate NL from PL1+PL2, validate via regeneration, flag `nl_valid`

Output: `extended/unified_dataset.jsonl`

## Project layout

```
codgen/
├── config.py
├── week1_setup_and_baselines.ipynb
├── lib/
│   ├── benchmarks.py      # dataset loaders
│   ├── models.py          # Qwen loaders
│   ├── prompts.py         # prompt builders
│   ├── execution.py       # Python/Java sandboxes
│   ├── ast_compare.py     # validation helpers
│   └── extend.py          # run_extension()
└── results/
    └── baselines.csv
```
