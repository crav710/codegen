# codgen — AIML PGCP Capstone (CodeGen / Task 3)

**Repo:** https://github.com/crav710/codgen  
(Team fork of https://github.com/svamsikrishna54/codgen)

## Branches

| Branch | Contents |
|--------|----------|
| `main` | Week 1 K=0 baselines only |
| `mentor-extension` | Baselines + mentor dataset extension (`run_extension`) |

## Colab quick start

1. Open `week1_setup_and_baselines.ipynb` in Google Colab
2. Runtime → **T4 GPU**
3. Run **Part 1** (cell 1.1 clones/pulls branch `mentor-extension` from `crav710/codgen`)
4. Run **Part 2** for baselines (optional)
5. Run **Part 3** for mentor dataset extension

## Push to your GitHub (one-time)

```bash
# 1. Create empty repo on GitHub: https://github.com/new  → name: codgen
# 2. Then from this folder:
git remote add mygithub https://github.com/crav710/codgen.git   # skip if already added
git push -u mygithub mentor-extension
git push mygithub main   # optional: also push main
```

## Mentor task (Part 3)

Extends each benchmark row to **NL + PL1 (Python) + PL2 (Java)**:

- **HumanEval / MBPP:** generate PL2 from NL+PL1, validate, flag `pl2_valid`
- **HumanEval-X:** generate NL from PL1+PL2, validate via regeneration, flag `nl_valid`

Output: `extended/unified_dataset.jsonl`
