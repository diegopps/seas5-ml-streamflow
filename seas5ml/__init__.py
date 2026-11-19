"""
seas5ml – Shared library for the SEAS5-driven seasonal streamflow pipeline.

The stage scripts under pipeline/ put the repository root on sys.path and
assemble their runs from these modules:

    paths             every input and output location (reads config.toml)
    catchments        catchment definitions (add new catchments here)
    data              discharge + feature loading and assembly
    climatology       smoothed DOY climatology, fit per training run
    features          dynamic feature set, a0, flat-table build
    designs           evaluation designs: hydro years -> folds (loyo)
    metrics           NSE / KGE / ACC / fair CRPS
    evaluate          ensemble-mean collapse and scoring
    io_results        stage-1 output files
    tuning            stage-1 tuned variant: nested-CV search for one fold
    sensitivity       stage-2 donor-year forcing swap

    models            fit / predict / save / load for every learned model
      configs         default configurations and tuning grids
      tabular         Ridge / RF / XGB on the flat table
      lstm            the LSTM
      references      climatology / persistence / damped-persistence baselines

    plots
      common          shared plot configuration, loaders and statistics
      boxplots        default-run skill boxplots per lead-time window
      heatmap         ACC by initialisation month x lead-time bin
      configrank      tuned-run config ranking table
      tuningvsref     tuned vs. default boxplots
      sensitivity     stage-2 forcing-sensitivity strip plots
"""
