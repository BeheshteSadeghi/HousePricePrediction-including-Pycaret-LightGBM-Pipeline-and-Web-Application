# Tehran Home Listing Price Estimator

Fits a LightGBM model to historical Tehran listings and presents an interactive property estimator.

[Run the project](#quick-start) · [Verification](#verification) · [Files](#repository-map)

| | |
| --- | --- |
| Stack | LightGBM · Streamlit · scikit-learn |
| Entry point | [`pricing.py`](pricing.py) |
| Focus | real-estate · lightgbm · streamlit · regression |

## How it works

`Listings → clean features → fit on log prices → holdout evaluation → Streamlit demo`

## Quick start

```sh
python -m pip install -r req.txt
python pricing.py
streamlit run webApp.py
```

The app trains and caches the pipeline using the bundled listing sample. The benchmark uses a separate 20% holdout and compares LightGBM with the training median. Neighborhood encoding is fit on the training split, and the model predicts log price before converting back to the original price units.

The [listing dataset](housePrice.csv) was collected from Divar. The target is the supplied `Price` column. These are historical asking prices, and the holdout is a random sample of those listings. The estimate depends on how well a property is represented in that sample.

[Exploration and model comparison](HousePricePrediciton_final.ipynb) · [Original preprocessing pipeline](PipelineModel.ipynb)

## Verification

Median absolute error was about 632 million for LightGBM versus 2.0 billion for the training-median baseline in the supplied price units.

## Repository map

| Path | Purpose |
| --- | --- |
| [`pricing.py`](pricing.py) | Estimator and benchmark |
| [`webApp.py`](webApp.py) | Interactive UI |
| [`housePrice.csv`](housePrice.csv) | Listing data |
| [`PipelineModel.ipynb`](PipelineModel.ipynb) | Earlier pipeline study |

## Scope

The data describes asking prices from a historical snapshot. The split is random and does not measure transfer to a new period.
