from pathlib import Path

import json

import pandas as pd


RESULTS_DIR = Path(
    "results"
)


def save_results(
    results
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = (
        pd.DataFrame(
            results
        )
    )

    csv_path = (
        RESULTS_DIR /
        "all_results.csv"
    )

    json_path = (
        RESULTS_DIR /
        "all_results.json"
    )

    excel_path = (
        RESULTS_DIR /
        "GAT_Research_Results.xlsx"
    )

    # ========================================================
    # CSV
    # ========================================================

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # JSON
    # ========================================================

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # ========================================================
    # EXCEL
    # ========================================================

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        dataframe.to_excel(
            writer,
            sheet_name="All Results",
            index=False,
        )

        # Dataset sheets
        for dataset_name in (
            dataframe[
                "Dataset"
            ].unique()
        ):

            subset = dataframe[
                dataframe[
                    "Dataset"
                ]
                ==
                dataset_name
            ]

            subset.to_excel(
                writer,
                sheet_name=
                    dataset_name[:31],
                index=False,
            )

        # Model average
        model_average = (
            dataframe
            .groupby("Model")
            [
                [
                    "Validation Accuracy",
                    "Validation Macro F1",
                    "Test Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1",
                    "Weighted F1",
                ]
            ]
            .mean()
            .reset_index()
        )

        model_average.to_excel(
            writer,
            sheet_name="Model Average",
            index=False,
        )

        # Dataset average
        dataset_average = (
            dataframe
            .groupby("Dataset")
            [
                [
                    "Validation Accuracy",
                    "Validation Macro F1",
                    "Test Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1",
                    "Weighted F1",
                ]
            ]
            .mean()
            .reset_index()
        )

        dataset_average.to_excel(
            writer,
            sheet_name="Dataset Average",
            index=False,
        )

        # Best model by test accuracy
        best_test = (
            dataframe.loc[
                dataframe
                .groupby(
                    "Dataset"
                )[
                    "Test Accuracy"
                ]
                .idxmax()
            ]
            .reset_index(
                drop=True
            )
        )

        best_test.to_excel(
            writer,
            sheet_name="Best Test",
            index=False,
        )

        # Best model by validation F1
        best_val_f1 = (
            dataframe.loc[
                dataframe
                .groupby(
                    "Dataset"
                )[
                    "Validation Macro F1"
                ]
                .idxmax()
            ]
            .reset_index(
                drop=True
            )
        )

        best_val_f1.to_excel(
            writer,
            sheet_name="Best Validation F1",
            index=False,
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"CSV   : {csv_path}"
    )

    print(
        f"JSON  : {json_path}"
    )

    print(
        f"Excel : {excel_path}"
    )