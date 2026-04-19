# State of the Field — Classical ML (2025)

## Where Classical ML Still Wins

Despite the rise of deep learning, classical ML — especially gradient boosting — dominates **tabular data** tasks. In Kaggle competitions and industry pipelines, XGBoost/LightGBM/CatBoost are the go-to choices for structured data.

Key insight: for tabular tasks with <1M rows and meaningful feature engineering, tree-based methods frequently outperform neural networks.

## Benchmark Leaders (Tabular Data)

- **XGBoost / LightGBM / CatBoost** — still the standard on most tabular benchmarks
- **TabNet (Google, 2021)** — attention-based neural net for tabular data; competitive but not dominant
- **TabPFN (2022)** — transformer trained to do in-context learning on small tabular datasets; surprisingly strong for n<1000
- **AutoML frameworks**: AutoGluon, H2O AutoML, FLAML — often ensemble tree models and light neural nets

## Key Papers

1. **"Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022, NeurIPS)**
   - Systematic comparison; trees win because tabular data has irregular decision boundaries and uninformative features that trees handle better
2. **"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second" (Hollmann et al., 2022)**
   - Prior-fitted networks; meta-learning for tabular tasks
3. **"Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021, NeurIPS)**
   - FT-Transformer: feature tokenizer + Transformer; good strong baseline

## Active Research Directions

- **Tabular foundation models**: Can LLM-style pretraining on diverse tabular data yield strong zero-shot models?
- **Feature interaction learning**: Automatic feature engineering via neural nets
- **Uncertainty quantification**: Conformal prediction, Bayesian methods for production safety
- **Federated ML**: Training across distributed, private data sources

## Useful Resources

- [Kaggle Tabular Playground Series](https://www.kaggle.com/competitions) — real datasets, community solutions
- [OpenML](https://www.openml.org/) — benchmark suite for classical ML
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — best reference for implementations
- Blog: *"Winning with XGBoost"* series on Towards Data Science
