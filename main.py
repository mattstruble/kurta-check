import pandas as pd

from hungarian import assign_suits

PENALTY = 20
NAME_COL = "Name"


TO_DROP = [NAME_COL]
PENALTY_COLS = {"Shoulder", "Chest", "Sleeve"}

if __name__ == "__main__":
    groomsmen_df = pd.read_csv("men.csv")
    suits_df = pd.read_csv("kurtas.csv")

    pre_assignment = {5: 5}

    suit_assignments, cost_matrix = assign_suits(
        groomsmen_df=groomsmen_df,
        suits_df=suits_df,
        name_col=NAME_COL,
        drop_cols=TO_DROP,
        pre_assignments=pre_assignment,
        penalty_cols=PENALTY_COLS,
        penalty=30,
    )

    print(cost_matrix)
    print(suit_assignments)

    suit_assignments.to_csv("suit_assignments.csv", index=False)
