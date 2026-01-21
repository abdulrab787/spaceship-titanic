import pandas as pd

def preprocess(df):
    df = df.copy()

    # Split Cabin
    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)

    # Total spending
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].fillna(0).sum(axis=1)

    # Group size
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")

    # Drop unused columns
    df.drop(["Name", "Cabin", "PassengerId"], axis=1, inplace=True)

    return df
