import pandas as pd
from scipy.optimize import linear_sum_assignment

PENALTY = 20
NAME_COL = "Name"


TO_DROP = [NAME_COL]
PENALTY_COLS = {"Shoulder", "Chest", "Sleeve"}


def assign_suits(groomsmen_df, suits_df):
    cost_matrix = []

    common_columns = groomsmen_df.columns.intersection(suits_df.columns).tolist()
    for col in TO_DROP:
        common_columns.remove(col)

    for _, groomsman in groomsmen_df.iterrows():
        groomsman_measurements = groomsman[common_columns]
        differences = suits_df[common_columns]

        cost_row = []

        for index, suit in differences.iterrows():
            cost = 0
            for measurement in common_columns:
                if (
                    suit[measurement] < groomsman_measurements[measurement]
                    and measurement in PENALTY_COLS
                ):
                    # Apply penalty if the suit is too small
                    cost += PENALTY + abs(
                        suit[measurement] - groomsman_measurements[measurement]
                    )
                else:
                    cost += abs(suit[measurement] - groomsman_measurements[measurement])
            cost_row.append(cost)

        cost_matrix.append(cost_row)

    cost_matrix = pd.DataFrame(cost_matrix, columns=suits_df[NAME_COL])
    print(cost_matrix.head(10))

    # Use the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = pd.DataFrame(
        {
            "Groomsman": groomsmen_df[NAME_COL].iloc[row_ind].values,
            "Suit": suits_df[NAME_COL].iloc[col_ind].values,
            "Difference": cost_matrix.values[row_ind, col_ind],
        }
    )

    return assignments


if __name__ == "__main__":
    groomsmen_df = pd.read_csv("men.csv")
    suits_df = pd.read_csv("kurtas.csv")

    suit_assignments = assign_suits(groomsmen_df, suits_df)

    print(suit_assignments)

    suit_assignments.to_csv("suit_assignments.csv", index=False)
