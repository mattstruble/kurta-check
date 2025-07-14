import pandas as pd
from scipy.optimize import linear_sum_assignment


def assign_suits(
    groomsmen_df: pd.DataFrame,
    suits_df: pd.DataFrame,
    name_col: str,
    drop_cols: list[str] | None = None,
    penalty_cols: set[str] = None,
    pre_assignments: dict[int, int] = None,
    penalty: int = 100,
):
    drop_cols = drop_cols if drop_cols else []
    penalty_cols = penalty_cols if penalty_cols else {}
    pre_assignments = pre_assignments if pre_assignments else {}

    cost_matrix = []

    common_columns = groomsmen_df.columns.intersection(suits_df.columns).tolist()

    for col in drop_cols:
        common_columns.remove(col)

    for groomsman_index, groomsman in groomsmen_df.iterrows():
        groomsman_measurements = groomsman[common_columns]
        differences = suits_df[common_columns]

        cost_row = []

        for suit_index, suit in differences.iterrows():
            cost = 0
            for measurement in common_columns:
                if (
                    suit[measurement] < groomsman_measurements[measurement]
                    and measurement in penalty_cols
                ):
                    # Apply penalty if the suit is too small
                    cost += penalty + abs(
                        suit[measurement] - groomsman_measurements[measurement]
                    )
                else:
                    cost += abs(suit[measurement] - groomsman_measurements[measurement])

            if (
                pre_assignments
                and groomsman_index in pre_assignments
                and pre_assignments[groomsman_index] == suit_index
            ):
                cost = 0
            cost_row.append(cost)

        cost_matrix.append(cost_row)

    cost_matrix = pd.DataFrame(cost_matrix, columns=suits_df[name_col])

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = pd.DataFrame(
        {
            "Groomsman": groomsmen_df[name_col].iloc[row_ind].values,
            "Suit": suits_df[name_col].iloc[col_ind].values,
            "Difference": cost_matrix.values[row_ind, col_ind],
        }
    )

    return assignments, cost_matrix
