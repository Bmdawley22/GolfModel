import sys
import argparse
import pandas as pd
import glob

# Function to get the PGA schedule for a given year and return the list of tourney names in order of occurrence


def load_current_scoring_csv_data(year):
    # Find CSV file that ends with f"{year}_scoring.csv"
    files = glob.glob(f"*{year}_scoring.csv")
    if not files:
        raise FileNotFoundError(f"No CSV file found for year {year}.")

    csv_file = files[0]  # Assume only one match
    print(f"Loading data from {csv_file}...")

    # Read the CSV
    df = pd.read_csv(csv_file)

    # Extract first TOURNAMENT
    tourney = df.loc[0, "TOURNAMENT"]

    # Find tourney in SCHEDULE and get the previous value
    schedule_series = df["SCHEDULE"]
    matching_indices = schedule_series[schedule_series == tourney].index

    if len(matching_indices) == 0:
        raise ValueError(f"Tourney '{tourney}' not found in SCHEDULE column.")

    tourney_index = matching_indices[0]

    if tourney_index == 0:
        ytd_tourney = None
    else:
        ytd_tourney = schedule_series.iloc[tourney_index - 1]

    avg_scoring = df["AVG_SCORE"]

    print(tourney)
    print(ytd_tourney)
    print(avg_scoring)

    return tourney, ytd_tourney, avg_scoring


def main():
    parser = argparse.ArgumentParser(description="PGA Tour Model")
    parser.add_argument("year")
    # parser.add_argument("num_past_tourneys")

    args = parser.parse_args()

    try:
        tourney, ytd_tourney, avg_scoring = load_current_scoring_csv_data(
            args.year)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
