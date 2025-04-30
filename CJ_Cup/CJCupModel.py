import sys
import argparse
import pandas as pd
import glob
import numpy as np

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

    players = df["PLAYER"]
    avg_scoring = df["AVG_SCORE"]

    print(tourney)
    print(avg_scoring)

    return tourney, players, avg_scoring


def load_all_prev_avg_scoring_normalized(model_year, players):
    # Find all CSV files in the current directory containing "_scoring"
    all_scoring_files = glob.glob("*_scoring*.csv")

    # Filter out files containing "{model_year}_scoring"
    prev_tourney_files = [
        file for file in all_scoring_files if f"{model_year}_scoring" not in file]

    if not prev_tourney_files:
        print(
            f"No CSV files found containing '_scoring' (excluding '{model_year}_scoring').")
        # Return an empty DataFrame with just the PLAYERS column
        return pd.DataFrame({'PLAYERS': players})

    # Create a DataFrame with the PLAYERS column
    result_df = pd.DataFrame({'PLAYERS': players})

    # Loop through each year and see if there is a file for each year
    for year in range(2005, int(model_year)):
        file_found = [file for file in prev_tourney_files if str(year) in file]
        if file_found:
            print(file_found[0])

            try:
                # Read the CSV
                df = pd.read_csv(file_found[0])

                # Check if required columns exist
                if "AVG_SCORE_Z" not in df.columns or "PLAYER" not in df.columns:
                    print(
                        f"Warning: 'AVG_SCORE_Z' or 'PLAYER' column not found in {file_found[0]}. Skipping file.")
                    # result_df[str(year)] = -0.5
                    continue

                # Create a Series mapping players to their AVG_SCORE_Z values
                avg_score_z_dict = df.set_index(
                    'PLAYER')['AVG_SCORE_Z'].to_dict()

                # For each player, get their AVG_SCORE_Z or set to -0.5 if not found
                avg_score_z = []
                for player in players:
                    if player in avg_score_z_dict:
                        # Ensure float
                        avg_score_z.append(
                            round(float(avg_score_z_dict[player]), 4))
                    else:
                        avg_score_z.append(-0.5)

                # Add the scores as a new column for this year
                result_df[str(year)] = avg_score_z
            except Exception as e:
                print(
                    f"Error processing file {file_found[0]} for year {year}: {e}")
                result_df[str(year)] = -0.5
    print("\nPrevious tournament AVG_SCORE_Z DataFrame:")
    print(result_df)
    return result_df


def compute_weighted_avg_scores(past_scores_df, model_year):
    # Get the list of years (columns except 'PLAYERS')
    years = [int(col) for col in past_scores_df.columns if col != 'PLAYERS']

    print(f"Years in table: {years}")

    if not years:
        print("No past scoring data available. Setting weighted average to 0.")
        return pd.Series(0, index=past_scores_df.index)

    # Compute weights using exponential decay
    lambda_decay = 0.2  # Decay rate
    weights = []
    for year in years:
        time_diff = (int(model_year) - 1) - year
        weight = np.exp(-lambda_decay * time_diff)
        weights.append(weight)

    # Print the weights for debugging
    print("\nWeights for each year before normalization:")
    for year, weight in zip(years, weights):
        print(f"Year {year}: {weight:.4f}")

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        print("Total weight is 0. Setting weighted average to 0.")
        return pd.Series(0, index=past_scores_df.index)
    weights = [w / total_weight for w in weights]

    # Print the weights for debugging
    print("\nNormalized weights for each year:")
    for year, weight in zip(years, weights):
        print(f"Year {year}: {weight:.4f}")

    # Compute the weighted average for each player
    scores_matrix = past_scores_df.drop(columns=['PLAYERS']).to_numpy()
    weighted_avg = np.average(scores_matrix, axis=1, weights=weights)

    # Add the weighted averages as a new column to past_scores_df
    past_scores_df['Weighted_Avg_Past_Score'] = weighted_avg

    print("\nUpdated past_scores_df with weighted averages:")
    print(past_scores_df)

    return past_scores_df


def main():
    parser = argparse.ArgumentParser(description="PGA Tour Model")
    parser.add_argument("model_year")

    args = parser.parse_args()

    try:
        tourney, players, avg_scoring_to_model = load_current_scoring_csv_data(
            args.model_year)

        past_scores_norm_df = load_all_prev_avg_scoring_normalized(
            args.model_year, players)

        past_scores_z_wtd_avg_df = compute_weighted_avg_scores(
            past_scores_norm_df, args.model_year)

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
