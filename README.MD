## Spaceship titanic
Solution of Kaggle problem: www.kaggle.com/competitions/spaceship-titanic


## Metrics (Logistic Regression Model)
- `Accuracy:` 0.7301 -> `0.7331`
- `Precision:` 0.7687 -> `0.7694`
- `Recall:` 0.6522 -> `0.6600`
- `F1-Score:` `0.7105`
- `ROC, AUC score:` `0.7326`

## Stack
- Python
- Scikit-learn
- Pandas
- Numpy

## EDA (Marimo notebook)
- Removed Passenger's `Name` and `Id`
- Extracted `Deck` letter from cabin
- Summarized `Spendings` into one column


## Solution
- Encoded categorical columns to `one hot` ones.
- Used `Pipeline`, `ColumnTransformer`, `FunctionTransformer`, from `sklearn`  
- Split data into `test` and `train` dataframes 
- Splined `Age` to best match with logistic regression

## Quick Start

`!!! I'm using kaggle for data downloading. If you want to download data you should:`
1) Create kaggle account
2) Join competition https://www.kaggle.com/competitions/spaceship-titanic
3) In settings -> api generate api key
4) Create .env file
5) Copy your KAGGLE_API_TOKEN=.... to .env

``` bash
git clone https://github.com/bagomedd/space-titanic
cd space-titanic

touch .env # copy your KAGGLE_API_TOKEN to .env

### using uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
### using pip
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .

### starts
python3 -m .src.train
```

## Important
- Marimo notebook doesn't install automatically via requirements.txt. To run `research.marimo.py` install it manually:
``` bash
### uv 
uv pip install marimo 
### pip
pip install marimo

### start 
marimo edit eda.marimo.py
```

