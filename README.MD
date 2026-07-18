## Spaceship titanic
Solution of Kaggle problem: www.kaggle.com/competitions/spaceship-titanic


## Metrics (Logistic Regression Model)
- Accuracy: 0.7301
- Precision: 0.7687
- Recall: 0.6522

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
- Splitted data to `test` and `train` dataframes 

## Start
``` bash
git clone https://github.com/bagomedd/space-titanic
cd space-titanic
### using uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

### using pip
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

### start
python3 main.py
```

## Important
- Marimo notebook doesn't install automatically via requirements.txt. To run `research.marimo.py` install it manually:
``` bash
### uv 
uv pip install marimo 
### pip
pip install marimo

### start 
marimo edit research.marimo.py
```

