# Tehran Housing Listing Price Estimator

A LightGBM regression project with a Streamlit interface. Estimate an asking price using floor area, neighborhood, rooms, and amenities.

```sh
python -m pip install -r req.txt
python pricing.py
streamlit run webApp.py
```

The app trains and caches the pipeline using the bundled listing sample. The benchmark uses a separate 20% holdout and compares LightGBM with the training median. Neighborhood encoding is fit on the training split, and the model predicts log price before converting back to the original price units.

The [listing dataset](housePrice.csv) was collected from Divar. The target is the supplied `Price` column. These are historical asking prices, and the holdout is a random sample of those listings. The estimate depends on how well a property is represented in that sample.

[Exploration and model comparison](HousePricePrediciton_final.ipynb) · [Original preprocessing pipeline](PipelineModel.ipynb)

## Example result

In the bundled holdout, median absolute error was about **632 million** for LightGBM and **2.0 billion** for the training-median baseline, in the source price units.
