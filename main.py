import pandas as pd
from scipy.optimize import linear_sum_assignment

NAME_COL = "Name"
LENGTH_COL = "Length"

groomsmen_df = pd.read_csv("men.csv")
suits_df = pd.read_csv("kurtas.csv")

TO_DROP = [NAME_COL]


def assign_suits(groomsmen_df, suits_df):
    cost_matrix = []

    common_columns = groomsmen_df.columns.intersection(suits_df.columns).tolist()
    for col in TO_DROP:
        common_columns.remove(col)

    for _, groomsman in groomsmen_df.iterrows():
        groomsman_measurements = groomsman[common_columns]
        differences = (
            suits_df[common_columns].sub(groomsman_measurements.values, axis=1).abs()
        )
        total_differences = differences.sum(axis=1)
        cost_matrix.append(total_differences.values)

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
    suit_assignments = assign_suits(groomsmen_df, suits_df)

    print(suit_assignments)

    suit_assignments.to_csv("suit_assignments.csv", index=False)
