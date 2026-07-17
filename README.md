# Fitness Tracker — Barbell Exercise Classification from Wearable Sensor Data

A machine learning pipeline that classifies barbell exercises (and detects reps)
from raw accelerometer and gyroscope data recorded by a wrist-worn IMU sensor
(MbientLab MetaMotion). The project goes end-to-end: raw sensor CSVs →
cleaning → feature engineering → model comparison.

## What it does

Given time-series accelerometer (x, y, z) and gyroscope (x, y, z) readings, the
pipeline predicts which of five barbell exercises was performed:

- Bench press
- Deadlift
- Overhead press
- Barbell row
- Squat

...plus a "rest" category for non-exercise periods.

## Data

- **Source:** 187 raw CSV files under `data/raw/MetaMotion/`, exported from a
  MetaMotion sensor (accelerometer @ 12.5 Hz, gyroscope @ 25 Hz).
- **Participants:** 5 (labeled A–E in the filenames).
- **Labels:** encoded directly in each filename, e.g.
  `A-bench-heavy2-rpe8_MetaWear_...csv` → participant `A`, exercise `bench`,
  category `heavy2` (weight/set descriptor, sometimes with an RPE — rate of
  perceived exertion — rating).
- Each exercise "set" was recorded separately, so both sensors' streams are
  reconstructed per set and merged.

## Methodology / pipeline

The pipeline is implemented as a sequence of script-based "notebooks" (cell
markers `# ---`) under `src/`, each reading the previous stage's pickle and
writing the next one to `data/interim/`:

1. **`src/data/make_dataset.py`** — parses participant/exercise/category out of
   each filename, merges the accelerometer and gyroscope streams (different
   sampling rates), and resamples everything to a common 200 ms grid
   (mean aggregation), grouped by day to avoid bridging gaps between
   recording sessions. Output: `01_data_processed.pkl`.

2. **`src/features/remove_outliers.py`** — compares three outlier-detection
   approaches: IQR, **Chauvenet's criterion**, and Local Outlier Factor (LOF).
   The Chauvenet-based result is the one exported and carried forward
   (`02_outlier_removed_chauvenet.pkl`). The Chauvenet and LOF implementations
   are adapted from the [ML4QS reference code](https://github.com/mhoogen/ML4QS)
   (credited in-line in the source).

3. **`src/features/build_features.py`** — builds the modeling feature set:
   - linear interpolation for missing values,
   - a Butterworth low-pass filter to denoise each sensor axis,
   - PCA (3 components) over the six raw axes,
   - scalar magnitude features (`acc_r`, `gyr_r`, the Euclidean norm of each
     sensor's x/y/z),
   - temporal abstraction (rolling mean/std per axis),
   - frequency-domain features via a Fourier transformation (per-set, to
     avoid leaking across exercise-set boundaries),
   - a KMeans cluster label (k=5) over the acceleration axes as an extra
     categorical feature.
   Output: `03_data_feature.pkl`.

4. **`src/models/train_model.py`** — the modeling stage:
   - stratified 75/25 train/test split,
   - forward feature selection (decision tree, up to 10 features) to find a
     compact, high-signal feature subset,
   - grid search across **5 feature sets** (raw axes only; raw + PCA/magnitude;
     + temporal; + frequency/cluster; forward-selected) × **5 classifiers**
     (feedforward neural network, random forest, KNN, decision tree, naive
     Bayes), comparing test accuracy for each combination,
   - a second evaluation that holds out **one entire participant** (all of
     participant A's sets) as the test set instead of a random split, to check
     whether the model generalizes to a person it never saw during training
     rather than just to unseen reps from people it has seen.

## Results

The training script (`src/models/train_model.py`) produces accuracy
comparisons and confusion matrices, but it renders them interactively via
`matplotlib` (`plt.show()`) — it does not persist accuracy numbers, a
trained model file, or a metrics report anywhere in the repository. There are
currently no committed results (numbers, plots, or serialized models) to cite,
so none are claimed here. To see actual numbers, run the pipeline yourself
(see below) — the script will print progress to the console and display the
accuracy-comparison bar chart and confusion matrices as it runs.

`reports/figures/` contains exploratory plots (raw sensor traces per exercise
per participant, e.g. `Bench (A).png`) generated during development, not
model-evaluation output.

## Repository structure

```
data/
  raw/MetaMotion/        raw per-set sensor CSVs + a MetaMotion.zip archive
  interim/                pickled intermediate datasets (01–03, see pipeline above)
reports/figures/          exploratory sensor plots per exercise/participant
src/
  data/make_dataset.py            stage 1: load, merge, resample
  features/
    remove_outliers.py            stage 2: outlier detection/removal
    build_features.py             stage 3: filtering, PCA, temporal/frequency features, clustering
    DataTransformation.py         low-pass filter + PCA helpers
    TemporalAbstraction.py        rolling-window aggregation helper
    FrequencyAbstraction.py       Fourier-feature helper
  models/
    train_model.py                stage 4: feature/model comparison + evaluation
    LearningAlgorithms.py         classifier wrappers (NN, RF, KNN, DT, NB) with grid search
    predict_model.py              🚧 empty — no inference/serving script yet
environment.yml           partial conda environment (see note below)
```

`references/folder_structure.txt` documents an aspirational
[cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
layout (`Makefile`, `setup.py`, `docs/`, `models/`, `data/processed`, etc.);
several of those paths don't exist yet in this repo — the structure above
reflects what's actually here.

## Setup

```bash
conda env create -f environment.yml
conda activate tracking-barbell-exercises
```

**Note:** `environment.yml` is incomplete and `requirements.txt` is empty —
the scripts also import `scikit-learn`, `scipy`, and `seaborn`, which aren't
listed in either file. Install them manually:

```bash
pip install scikit-learn scipy seaborn
```

The scripts use relative paths (e.g. `../../data/raw/...`) and are written as
notebook-style cells, so run them from within their own directory
(`src/data/`, `src/features/`, `src/models/`) — e.g. via VS Code/Jupyter cell
execution — rather than as standalone scripts from the repo root, and run
them in pipeline order (`make_dataset.py` → `remove_outliers.py` →
`build_features.py` → `train_model.py`), since each stage depends on the
previous stage's pickle output.

## Known limitations / roadmap

- 🚧 `predict_model.py` is empty — there's no script to load a trained model
  and classify new sensor data.
- 🚧 No trained model is serialized/saved; every run retrains from scratch.
- 🚧 `requirements.txt` is empty and `environment.yml` is missing several
  used packages (see Setup).
- 🚧 No automated tests.
- 🚧 No license file.

## Acknowledgments

The Chauvenet's-criterion and Local Outlier Factor outlier-detection
functions in `src/features/remove_outliers.py` are adapted from the
[ML4QS ("Machine Learning for the Quantitative Self") reference
implementation](https://github.com/mhoogen/ML4QS), as credited in the source.
