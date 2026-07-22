import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():

    import marimo as mo
    from typing import cast
    import pandas 
    import seaborn as sns
    import matplotlib.pyplot as plt



    from sklearn.model_selection import train_test_split
    from sklearn.utils import Bunch
    from sklearn.inspection import permutation_importance
    from src.data_prepare import load_initial_df
    from src.log_model.log_pipeline import log_model_pipeline

    return (
        Bunch,
        cast,
        load_initial_df,
        log_model_pipeline,
        mo,
        permutation_importance,
        plt,
        sns,
        train_test_split,
    )


@app.cell
def _(load_initial_df, train_test_split):
    df = load_initial_df()

    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(
    Bunch,
    X_test,
    X_train,
    cast,
    log_model_pipeline,
    permutation_importance,
    y_test,
    y_train,
):

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
    return result_f1, result_precision, result_recall


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
def _(X_train, log_model_pipeline, y_train):
    preprocessed_df = log_model_pipeline[:-1].fit_transform(X_train, y_train)
    preprocessed_df
    return (preprocessed_df,)


@app.cell
def _(preprocessed_df):
    prep_X = preprocessed_df
    prep_corr_matrix_pearson = prep_X.corr(method="pearson")
    prep_corr_matrix_kendall = prep_X.corr(method="kendall")
    prep_corr_matrix_spearman = prep_X.corr(method="spearman")
    return (
        prep_corr_matrix_kendall,
        prep_corr_matrix_pearson,
        prep_corr_matrix_spearman,
    )


@app.cell
def _(prep_corr_matrix_kendall):
    prep_corr_matrix_kendall
    return


@app.cell
def _(prep_corr_matrix_pearson):
    prep_corr_matrix_pearson
    return


@app.cell
def _(prep_corr_matrix_spearman):
    prep_corr_matrix_spearman
    return


@app.cell
def _(plt, prep_corr_matrix_kendall, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(prep_corr_matrix_kendall, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    return


@app.cell
def _(plt, prep_corr_matrix_spearman, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(prep_corr_matrix_spearman, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    return


@app.cell
def _(plt, prep_corr_matrix_pearson, sns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(prep_corr_matrix_pearson, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
