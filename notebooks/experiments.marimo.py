import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from typing import cast
    from sklearn.model_selection import train_test_split
    from sklearn.utils import Bunch
    from sklearn.inspection import permutation_importance
    from src.data_prepare import load_initial_df
    from src.pipeline import log_model_pipeline

    return (
        Bunch,
        cast,
        load_initial_df,
        log_model_pipeline,
        mo,
        permutation_importance,
        train_test_split,
    )


@app.cell
def _(load_initial_df):
    df = load_initial_df()
    return (df,)


@app.cell
def _(
    Bunch,
    cast,
    df,
    log_model_pipeline,
    permutation_importance,
    train_test_split,
):
    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)

    log_model_pipeline.fit(X_train, y_train)

    y_pred = log_model_pipeline.predict(X_test)

    result_f1 = cast(
        Bunch,
        permutation_importance(log_model_pipeline, X_test, y_test,scoring="f1", n_repeats=10, random_state=42)
    )
    result_precision = cast(
        Bunch,
        permutation_importance(log_model_pipeline, X_test, y_test,scoring="precision", n_repeats=10, random_state=42)
    )
    result_recall = cast(
        Bunch,
        permutation_importance(log_model_pipeline, X_test, y_test,scoring="recall", n_repeats=10, random_state=42)
    )

    return X_test, result_f1, result_precision, result_recall


@app.cell
def _(X_test, result_f1, result_precision, result_recall):
    print("\nF1 metrics\n")
    for i in result_f1.importances_mean.argsort():
        print(f"{X_test.columns[i]} : {result_f1.importances_mean[i]:.3f}")

    print("\nPrecision metrics\n")
    for i in result_precision.importances_mean.argsort():
        print(f"{X_test.columns[i]} : {result_precision.importances_mean[i]:.3f}")

    print("\nRecall metrics\n")
    for i in result_recall.importances_mean.argsort():
        print(f"{X_test.columns[i]} : {result_recall.importances_mean[i]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Home Planet имеет разные метрики важности по разным метрикам.
    - f1: HomePlanet : 0.012
    - precision: HomePlanet : 0.042
    - recall: HomePlanet : -0.012
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
