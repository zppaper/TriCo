# TriCo stage prediction

Run from the repository root:

```bash
python -m TriCo --data-dir /path/to/datasets
```

The data directory must contain:

```text
Stage_Prediction/
    COAD/{mRNA.csv,miRNA.csv,label.csv}
    KICH/{mRNA.csv,miRNA.csv,label.csv}
    KIRP/{mRNA.csv,miRNA.csv,label.csv}
    READ/{mRNA.csv,miRNA.csv,label.csv}
Stage_Prediction_gene_level/
    genes_anchor.txt
    COAD/cnv_gene.csv
    KICH/cnv_gene.csv
    KIRP/cnv_gene.csv
    READ/cnv_gene.csv
```

Molecular matrices contain features in rows and patient IDs in columns. Labels contain `sample_id` and binary `label` values (0 or 1). The optional `genes_anchor.txt` contains one gene symbol per line; when absent, the gene universe is derived from KEGG. Pathway annotations are read from `KEGG_pathways/20230205_kegg_hsa.gmt` in the repository root. Regulatory priors are included in `resources/relations/`.

This implementation covers COAD, KICH, KIRP, and READ stage prediction. The repository's LGG survival example is not an input to this entry point.

```bash
python -m TriCo --data-dir /path/to/datasets --datasets COAD --seeds 1028 --folds 1 --result-dir results/coad
python -m TriCo --help
```

Defaults are three seeds, five outer folds, an internal validation fraction of 0.20, and at most 120 training epochs. Standardization and supervised feature selection are fitted on the training subset. Model selection and modality reliability estimation use the internal validation subset.

| Module | Purpose |
| --- | --- |
| `config.py` | Paths and hyperparameters |
| `data.py` | Input alignment, projection, splitting, and scaling |
| `relations.py` | Regulatory relation matrices |
| `model.py` | Neural architecture |
| `training.py` | Losses, loaders, and optimization |
| `metrics.py` | ROC AUC and average precision |
| `evaluation.py` | Auxiliary predictors and prediction fusion |
| `results.py` | Result summaries and CSV output |
| `main.py` | Command-line arguments and experiment loop |

Outputs are `runs.csv`, `predictions.csv`, `summary.csv`, and `modality_balance_summary.csv`. The `aupr` column contains average precision. Summary standard deviations describe the selected seed-fold scores. Prediction `sample` values are indices within each test fold.

Required Python packages: NumPy, pandas, scikit-learn, and PyTorch. Install the repository dependencies with `pip install -r requirements.txt`.
