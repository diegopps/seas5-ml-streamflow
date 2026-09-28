"""
flowcast_src – Shared library for the SEAS5-driven seasonal streamflow pipeline.

The stage scripts in the repository root (00_preprocess_catchment.py,
01_exploration.py, 02_hyperparamtuning_nested.py,
03_sensitivity_climate_input.py and their *_plot_*.py counterparts) assemble
their runs from these modules:

    paths             input/output locations (reads config.toml)
    data              discharge + feature loading and assembly
    climatology       global smoothed DOY climatology
    features          dynamic feature set and flat-table build
    cv                LOGO-CV fold splitting
    metrics           NSE / KGE / ACC / CRPS
    evaluate          ensemble-mean collapse and scoring
    io_results        stage-01 output-directory layout and file writing
    references        climatology / persistence / damped-persistence baselines
    estimators        per-fold estimator factories for the tabular models
    tabular           Ridge / RF / XGB training loop
    lstm              LSTM model and training loop
    hptuning          stage-02 nested-CV search: config grids, train/score
    sensitivity       stage-03 donor-year forcing test and final-model fitting

    plot_lib          shared plot configuration, loaders and statistics
    plot_boxplots     stage-01 skill boxplots per lead-time window
    plot_heatmap      ACC by initialisation month x lead-time bin
    plot_configrank   stage-02 config ranking table
    plot_tuningvsref  stage-02 tuned vs. untuned boxplots
    plot_sensitivity  stage-03 forcing-sensitivity strip plots
"""
