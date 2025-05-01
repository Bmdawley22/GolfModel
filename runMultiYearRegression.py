import sys
import argparse
import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import subprocess


def find_earliest_year_and_years_to_model(tourney_folder):
    # Find all CSV files containing "_scoring.csv"
    scoring_files = glob.glob(f"{tourney_folder}/*_scoring.csv")
    if not scoring_files:
        raise FileNotFoundError(f"No scoring CSV files found in this folder.")

    # Extract years from filenames (e.g., Masters_Tournament_2010_scoring.csv → 2010)
    years_to_model = []
    for file in scoring_files:
        # Look for a 4-digit year in the filename
        parts = file.split("_")
        for part in parts:
            # Check if extracted file year is a valid, completed tournament year
            if part.isdigit() and len(part) == 4 and 2000 <= int(part) < 2025:
                years_to_model.append(int(part))
                break

    if not years_to_model:
        raise ValueError("No valid years found in scoring files.")

    earliest_year = min(years_to_model)
    # don't want to model the earliest year because there is no prev tourney data
    years_to_model.remove(earliest_year)

    print(f"Earliest year found: {earliest_year}")
    print(f"Years to model: {years_to_model}")
    return earliest_year, years_to_model


def load_scoring_data(year_to_model, tourney_folder):
    print(f"\nRunning load_scoring_data for year_to_model: {year_to_model}")
    # Find the scoring CSV file for the given year and tournament
    files = glob.glob(f"{tourney_folder}/*{year_to_model}_scoring.csv")
    if not files:
        raise FileNotFoundError(
            f"No scoring CSV file found for year: {year_to_model}.")

    csv_file = files[0]
    print(f"Loading scoring data from {csv_file}...")

    df = pd.read_csv(csv_file)
    if "PLAYER" not in df.columns or "AVG_SCORE" not in df.columns:
        raise ValueError(
            f"Required columns 'PLAYER' or 'AVG_SCORE' not found in {csv_file}.")

    players = df["PLAYER"]
    avg_scores = df["AVG_SCORE"]

    print(f"\n{csv_file} data loaded.")
    return players, avg_scores


def load_prev_scoring_z_and_calc_year_weights(earliest_year, year_to_model, years_to_model, tourney_folder):
    print(
        f"\nRunning load_prev_scoring_z_and_calc_year_weights for year_to_model: {year_to_model}")
    # Get all years before the current year we are modeling
    prev_years = [earliest_year]
    for year in years_to_model:
        if year < year_to_model:
            prev_years.append(year)
    if not prev_years:
        print(
            f"No previous years available for {years_to_model}. Setting weighted average to 0.")
        return None

    print(f"\nYears before current year_to_model: {prev_years}")

    print("\nLoad AVG_SCORE_Z data for each previous year...")
    # Load scoring data for each previous year
    past_scores = {}
    for year in prev_years:
        files = glob.glob(f"{tourney_folder}/*{year}_scoring.csv")
        if not files:
            continue

        df = pd.read_csv(files[0])
        if "PLAYER" not in df.columns or "AVG_SCORE_Z" not in df.columns:
            print(
                f"Warning: 'PLAYER' or 'AVG_SCORE_Z' not found in {files[0]}. Skipping.")
            continue

        past_scores[year] = df.set_index('PLAYER')['AVG_SCORE_Z'].to_dict()

    if not past_scores:
        print(
            f"No previous scoring data available for {year_to_model}. Setting weighted average to 0.")
        return None

    print("Previous year's AVG_SCORE_Z data loaded.")

    print("\nComputing previous years weights using exponential decay...")
    # Compute weights using exponential decay
    lambda_decay = 0.1  # Decay rate
    weights = []
    for year in prev_years:
        # Minus 1 because we want time_diff = 0 for the previous year
        time_diff = (year_to_model - 1) - year
        weight = np.exp(-lambda_decay * time_diff)
        weights.append(weight)

    print(f"\nWeights for years before {year_to_model}:")
    for year, weight in zip(prev_years, weights):
        print(f"Year {year}: {weight:.4f}")

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        print("Total weight is 0. Setting weighted average to 0.")
        return None
    weights = [w / total_weight for w in weights]

    print(f"\nNormalized weights for previous years before {year_to_model}:")
    for year, weight in zip(prev_years, weights):
        print(f"Year {year}: {weight:.4f}")

    return past_scores, weights


def compute_weighted_avg_scores(model_players, past_scores_z, year_weights):
    print("\nRunning compute_weighted_avg_scores...")
    if past_scores_z is None:
        return np.zeros(len(model_players))

    # Set up mapping between past_scores_z years and index
    past_scores_z_years = []
    for year in past_scores_z:
        past_scores_z_years.append(year)

    # Initialize weighted average for each player
    weighted_avg = np.zeros(len(model_players))
    # Loop throuh all players we are modeling
    for idx, player in enumerate(model_players):
        player_scores = []
        player_weights = []
        # Loop through all past results up to the current model year
        for year, scores in past_scores_z.items():
            player_weights.append(
                year_weights[past_scores_z_years.index(year)])
            if player in scores:  # Check if player played in tourney for current year
                # Get AVG_SCORE_Z for current player for each tourney
                player_scores.append(scores[player])
            else:
                player_scores.append(-0.1)  # Penalty for not playing

        # Compute weighted average for the player
        if player_weights:
            weighted_avg[idx] = np.average(
                player_scores, weights=player_weights)
            # if idx < 5: # For debugging
            #     print(player_scores)
            #     print(player_weights)
            #     print(np.round(weighted_avg[idx], 4))

    return np.round(weighted_avg, 4)


def load_ytd_data(year_to_model, tourney_folder):
    # Find YTD data for the year (earliest year + 1 for the first iteration)
    files = glob.glob(f"{tourney_folder}/*ytd_thru*{year_to_model}.csv")
    if not files:
        raise FileNotFoundError(
            f"No YTD CSV file found for year {year_to_model}.")

    csv_file = files[0]  # Assume one file for each year's ytd data
    print(f"Loading YTD data from {csv_file}...")

    df = pd.read_csv(csv_file)
    # print(f"YTD data for {year_to_model}:")
    # print(df.head())

    # Identify numeric columns (all except 'PLAYERS')
    numeric_cols = df.columns.drop('PLAYERS')

    # Columns to invert normalization (lower is better)
    invert_cols = ['Distance From Edge of Fairway', "Rough Tendency", "Fairway Proximity", "Approaches from > 200", "Proximity from Sand (Short)",
                   "Proximity from Rough (Short)", "Proximity from 30+", "Proximity ATG", "3-Putt Avoidance"]

    # Apply Z-score normalization
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:  # Avoid division by zero
            if col in invert_cols:
                df[col] = -1 * (df[col] - mean) / std
            else:
                df[col] = (df[col] - mean) / std
        else:
            df[col] = 0

    # Round normalized values to 4 decimal places
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def run_multi_year_regression(earliest_year, years_to_model, tourney_folder):
    combined_df = None

    # Iterate through each year from the year after the earliest_year to the latest completed tourney
    for idx, year_to_model in enumerate(sorted(years_to_model)):
        print(
            f"\n//////////////\nProcessing year to model {year_to_model}...\n//////////////\n")

        # Load scoring data for only the current year to model in the for loop
        model_players, scores_to_model = load_scoring_data(
            year_to_model, tourney_folder)

        # Load previous years' normalized scoring data and compute year weights
        past_scores_z, weights = load_prev_scoring_z_and_calc_year_weights(earliest_year,
                                                                           year_to_model, years_to_model, tourney_folder)
        # Computes each players yearly weighted AVG_SCORE_Z for previous tournaments based on current year_to_model
        weighted_avg_scores = compute_weighted_avg_scores(
            model_players, past_scores_z, weights)

        ytd_df = load_ytd_data(year_to_model, tourney_folder)

        # Create DataFrame for the current year
        year_df = pd.DataFrame({
            'PLAYERS': model_players,
            'Year_to_Model': year_to_model,
            'Avg_Score': scores_to_model,
            'Weighted_Avg_Past_Score_Z': weighted_avg_scores
        })

        # Merge with YTD data (inner merge to keep only players with YTD data)
        year_df = year_df.merge(ytd_df, on='PLAYERS', how='inner')

        # Stack the data
        if combined_df is None:
            combined_df = year_df
        else:
            combined_df = pd.concat(
                [combined_df, year_df], ignore_index=True)

    if combined_df is None or combined_df.empty:
        raise ValueError(
            "No data available after merging. Cannot run regression.")

    print("\nCombined DataFrame for all years:")
    print(combined_df)

    # Run linear regression on the combined data
    target_col = 'Avg_Score'
    feature_cols = [col for col in combined_df.columns if col not in [
        'PLAYERS', target_col, 'Year_to_Model']]

    # Prepare X and y
    X = combined_df[feature_cols]
    y = combined_df[target_col]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_imputed, y)

    # Print coefficients
    print("\nLinear Regression Coefficients:")
    for feature, coef in zip(feature_cols, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Save the coefficients to a CSV file
    weights_df = pd.DataFrame({
        'Feature': feature_cols + ['Intercept'],
        'Coefficient': list(model.coef_) + [model.intercept_]
    })
    weights_df['Coefficient'] = weights_df['Coefficient'].round(4)
    weights_filename = f"{tourney_folder}/multi_year_regression_weights.csv"
    weights_df.to_csv(weights_filename, index=False)
    print(f"Regression weights saved to {weights_filename}")

    # Extract coefficients and intercept
    intercept = weights_df[weights_df['Feature']
                           # Finds Intercept row and gets coefficient value
                           == 'Intercept']['Coefficient'].iloc[0]
    coefficients = weights_df[weights_df['Feature'] != 'Intercept'].set_index(
        'Feature')['Coefficient'].to_dict()

    # Prepare X (features) for prediction
    X = combined_df[feature_cols]

    # Handle missing values in features (X) by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    print("\nComputing score predictions...")

    # Compute predictions manually: intercept + sum(coef * feature)
    predictions = np.full(len(X_imputed), intercept, dtype=float)
    for idx, feature in enumerate(feature_cols):
        predictions += coefficients[feature] * X_imputed[:, idx]

    # Round predictions to 2 decimal places
    rounded_predictions = np.round(predictions, 2)

    combined_df.insert(loc=3, column='Predicted_Avg_Score',
                       value=rounded_predictions)

    # Save the combined DataFrame
    output_file = f"{tourney_folder}/multi_year_model.csv"
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="PGA Multi-Year Tournament Model")
    parser.add_argument('--predict-next-tourney', action='store_true',
                        help='Refresh data if this flag is present')

    args = parser.parse_args()

    tourney_folder = "CJ_Cup"

    try:
        # Find the earliest year that we have tournament stats and years we can model
        earliest_year, years_to_model = find_earliest_year_and_years_to_model(
            tourney_folder)

        # Run the multi-year regression
        run_multi_year_regression(
            earliest_year, years_to_model, tourney_folder)

        if args.predict_next_tourney:
            print("Running subprocess: python runSingleyearRegression.py 2025...")
            subprocess.run(["python", "runSingleYearRegression.py", "2025"])
            print("Predictions made for 2025.")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
