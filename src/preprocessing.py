import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Split Cabin
    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)

    # Convert CabinNum to numeric
    df["CabinNum"] = pd.to_numeric(df["CabinNum"], errors="coerce")

    # ===== Total Spend Feature =====
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df[spend_cols] = df[spend_cols].fillna(0)

    # Ensure spend columns are numeric
    for col in spend_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    # Passenger Group Features
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count").astype(int)

    # Fill Missing Values
    df["HomePlanet"].fillna("Unknown", inplace=True)
    df["Destination"].fillna("Unknown", inplace=True)
    df["VIP"] = df["VIP"].fillna(False)
    df["CryoSleep"].fillna(False, inplace=True)
    df["Age"].fillna(df["Age"].median(), inplace=True)

    # Drop Useless Columns
    df.drop(["Name", "Cabin", "PassengerId", "Group"], axis=1, inplace=True)

    return df