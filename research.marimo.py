import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import kagglehub
    import pandas as pd
    import marimo as mo
    from attr import dataclass
    from sklearn.model_selection import train_test_split

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.model_selection import train_test_split

    return kagglehub, mo, pd


@app.cell
def _(kagglehub, pd):
    path = kagglehub.competition_download('spaceship-titanic')

    sample_submission_df = pd.read_csv(path+"/sample_submission.csv") 
    # test_df = pd.read_csv(path+"/test.csv")
    df = pd.read_csv(path+"/train.csv") 

    train_df : pd.DataFrame; test_df : pd.DataFrame; val_df: pd.DataFrame; train_val_df: pd.DataFrame
    train_df = df
    train_df
    return df, train_df


@app.cell
def _(df, pd, train_df: "pd.DataFrame"):
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(df.info(), "\n-----\n")
        print(train_df.describe(include="all").T, "\n-----\n")
        print(df.isna().sum(), "\n-----\n")
        print(df.nunique(), "\n-----\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Home planet
    """)
    return


@app.cell
def _(df):
    print("Home planet")
    hp_df = df.groupby(["HomePlanet", "Transported"]).size().unstack()
    hp_df["Percentage"] = hp_df[True] / (hp_df[True] + hp_df[False])
    print(hp_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First letter of cabin
    """)
    return


@app.cell
def _(df):
    cabin_df = df.assign(
        Deck = lambda x : x["Cabin"].str[0].fillna("Unknown")
    )

    deck_analysis = (
        cabin_df.groupby("Deck")["Transported"]
        .agg(
            Total_Passengers="count",
            Survival_Rate="mean",
        )
        .reset_index()    
        .sort_values(by="Survival_Rate", ascending=False)
    ) 
    deck_analysis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Значит, палуба сильно влияет на выживаемост
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
