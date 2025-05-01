import sys
import argparse
import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to extract current year tourney data


def load_current_scoring_csv_data(year, tourney_completed, tourney_folder):
    # Find CSV file that ends with f"{year}_scoring.csv"
    files = glob.glob(f"{tourney_folder}/*{year}_scoring.csv")
    if not files:
        raise FileNotFoundError(f"No Scoring CSV file found for year {year}.")

    csv_file = files[0]  # Assume only one match
    print(f"Loading data from {csv_file}...")

    # Read the CSV
    df = pd.read_csv(csv_file)

    # Extract TOURNAMENT we are modeling
    tourney = df.loc[0, "TOURNAMENT"]
    # Extract the PLAYERs from the tourney we're modeling
    players = df["PLAYER"]

    print(f"Model tourney from csv: {tourney}")
    # Either predicting scores or modeling completed tournament
    if tourney_completed:
        # Tournament scores to model
        avg_scoring = df["AVG_SCORE"]
        print(f"Average scoring from csv: {avg_scoring}")
        odds_perc = None
    else:
        avg_scoring = None  # Tournament yet to occur
        betting_odds = df["ODDS TO WIN"]  # Odds for tournament
        print(betting_odds)

        # Convert betting odds to probabilities
        odds_perc = []
        for odds in betting_odds:
            probability = 100 / (100 + odds)
            odds_perc.append(np.round(100 * probability, 4))

    return tourney, players, avg_scoring, odds_perc


def load_all_prev_avg_scoring_normalized(model_year, players, tourney_folder):
    # Find all CSV files in the current directory containing "_scoring"
    all_scoring_files = glob.glob(f"{tourney_folder}/*_scoring*.csv")

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
    for year in range(2010, int(model_year)):
        file_found = [file for file in prev_tourney_files if str(
            year) in file]  # Check if year is in filename
        if file_found:
            print(file_found[0])
            try:
                # Read the current CSV
                df = pd.read_csv(file_found[0])

                # Check if required columns exist
                if "AVG_SCORE_Z" not in df.columns or "PLAYER" not in df.columns:
                    print(
                        f"Warning: 'AVG_SCORE_Z' or 'PLAYER' column not found in {file_found[0]}. Skipping file.")
                    continue

                # Create a Series mapping players to their AVG_SCORE_Z values
                avg_score_z_dict = df.set_index(
                    'PLAYER')['AVG_SCORE_Z'].to_dict()

                # For each player in model tournament, get their AVG_SCORE_Z or set to -0.1 if not found
                avg_score_z = []
                for player in players:
                    if player in avg_score_z_dict:
                        # Ensure float
                        avg_score_z.append(
                            round(float(avg_score_z_dict[player]), 4))
                    else:
                        # -0.1 is a slight penalty for player not playing in a past tournament
                        avg_score_z.append(-0.1)

                # Add the scores as a new column for this year
                result_df[str(year)] = avg_score_z
            except Exception as e:
                print(
                    f"Error processing file {file_found[0]} for year {year}: {e}")
                result_df[str(year)] = -0.1
    print("\nPrevious tournament AVG_SCORE_Z DataFrame:")
    print(result_df)
    return result_df


def compute_weighted_avg_scores(past_scores_df, model_year, odds_perc, tourney_completed):
    # Get the list of years (columns except 'PLAYERS')
    years = [int(col) for col in past_scores_df.columns if col != 'PLAYERS']

    print(f"\nYears in table: {years}")

    if not years:
        print("\nNo past scoring data available. Setting weighted average to 0.")
        return pd.Series(0, index=past_scores_df.index)

    # Compute weights using exponential decay for each year's data
    lambda_decay = 0.1  # Decay rate
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
        print("\nTotal weight is 0. Setting weighted average to 0.")
        return pd.Series(0, index=past_scores_df.index)
    weights = [w / total_weight for w in weights]

    # Print the weights for debugging
    print("\nNormalized weights for each year:")
    for year, weight in zip(years, weights):
        print(f"Year {year}: {weight:.4f}")

    # Compute the weighted average for each player
    # Drop PLAYERS column for weighted avg calc
    scores_matrix = past_scores_df.drop(columns=['PLAYERS']).to_numpy()

    if not tourney_completed:
        # Add Tourney Odds before weighted average column
        past_scores_df["Odds_Probability_%"] = odds_perc

    # Calculate weighted avg
    weighted_avg = np.average(scores_matrix, axis=1, weights=weights)

    # Add the weighted averages as a new column to past_scores_df
    past_scores_df['Weighted_Avg_Past_Score_Z'] = np.round(weighted_avg, 2)

    print("\nUpdated past_scores_df with weighted averages:\n")
    print(past_scores_df)

    # Remove all year columns (keep only 'PLAYERS' and 'Weighted_Avg_Past_Score_Z')
    year_columns = [col for col in past_scores_df.columns if col not in [
        'PLAYERS', 'Odds_Probability_%', 'Weighted_Avg_Past_Score_Z']]
    past_scores_df = past_scores_df.drop(columns=year_columns)

    # Display the updated DataFrame
    print("\nUpdated past_scores_df after removing year columns:")
    print(past_scores_df)

    return past_scores_df


def load_ytd_data_and_normalize(model_df, model_year, tourney_folder):
    # Find CSV file that ends with f"ytd_thru*{model_year}.csv"
    files = glob.glob(f"{tourney_folder}/ytd_thru*{model_year}.csv")
    if not files:
        raise FileNotFoundError(
            f"No YTD CSV file found for year {model_year}.")

    csv_file = files[0]  # Assume only one match
    print(f"\nLoading ytd data from {csv_file}...")

    # Read the CSV
    df = pd.read_csv(csv_file)

    print(f"\nData loaded from {csv_file}...")
    print(df.head())

    print("\nNormalizing data....")

    # Identify numeric columns (all except 'PLAYERS')
    numeric_cols = df.columns.drop('PLAYERS')

    # Columns to invert normalization (the smaller the data the more positive the stat)
    invert_cols = ['Distance From Edge of Fairway', "Rough Tendency", "Fairway Proximity", "Approaches from > 200", "Proximity from Sand (Short)",
                   "Proximity from Rough (Short)", "Proximity from 30+", "Proximity ATG", "3-Putt Avoidance"]

    # Apply Z-score normalization (stat - mean) / std dev
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:  # Avoid division by zero
            if col in invert_cols:
                # Invert Z-score for columns where lower is better
                df[col] = -1 * (df[col] - mean) / std
            else:
                # Standard Z-score for columns where higher is better
                df[col] = (df[col] - mean) / std
        else:
            # If standard deviation is 0, set normalized values to 0
            df[col] = 0

    # Round normalized values to 4 decimal places for readability
    df[numeric_cols] = df[numeric_cols].round(4)

    print("\nNormalized YTD DataFrame:")
    print(df.head())

    # Merge with model_df on 'PLAYERS' using an inner merge to keep only players with YTD data
    merged_df = model_df.merge(df, on='PLAYERS', how='inner')

    print("\nMerged DataFrame with YTD data (inner merge):")
    print(merged_df.head())

    return merged_df


def perform_linear_regression(avg_scoring_to_model, model_df, model_year, tourney, tourney_completed, tourney_folder):
    print("\nRunning perform_linear_regression function...")

    # Define the target variable (y) and features (X)
    # Note: if predicting a tourney there won't be an "...Avg_Score" column
    target_col = f"{model_year}_Avg_Score"
    feature_cols = [col for col in model_df.columns if col not in [
        # Get columns representing model features
        'PLAYERS', 'Odds_Probability_%', target_col]]

    # Check if we're performing a linear regression on completed tournament
    if tourney_completed:
        # Training mode: Fit a new linear regression model
        print("Training linear regression model on completed tournament...")

        model_df.insert(
            loc=1, column=f'{model_year}_Avg_Score', value=avg_scoring_to_model)

        # Drop rows where the target variable is NaN
        model_df_clean = model_df.dropna(subset=[target_col])

        if model_df_clean.empty:
            print("No valid data for regression after dropping NaN target values. Adding predicted column with zeros.")
            model_df.insert(loc=1, column='Predicted_Avg_Score', value=0)
            return model_df

        # Prepare X and y
        X = model_df_clean[feature_cols]
        y = model_df_clean[target_col]

        # Handle missing values in features (X) by imputing with the mean
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Fit the linear regression model
        model = LinearRegression()
        model.fit(X_imputed, y)

        # Predict on the entire dataset (including rows with NaN target, for completeness)
        X_full = model_df[feature_cols]
        X_full_imputed = imputer.transform(X_full)
        predictions = model.predict(X_full_imputed)

        # Round predictions to 2 decimal places
        rounded_predictions = np.round(predictions, 2)

        # Print the coefficients for debugging
        print("\nLinear Regression Coefficients:")
        for feature, coef in zip(feature_cols, model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {model.intercept_:.4f}")

        # Create a DataFrame with the coefficients and intercept
        weights_df = pd.DataFrame({
            'Feature': feature_cols + ['Intercept'],
            'Coefficient': list(model.coef_) + [model.intercept_]
        })

        # Round coefficients to 4 decimal places for readability
        weights_df['Coefficient'] = weights_df['Coefficient'].round(4)

        # Save the coefficients to a CSV file
        weights_filename = f"{tourney_folder}/{tourney}_{model_year}_weights.csv".replace(
            " ", "_")
        weights_df.to_csv(weights_filename, index=False)
        print(f"Regression weights saved to {weights_filename}")

    else:
        # Predict-only mode: Use pre-existing weights
        print("\nUsing pre-existing weights to predict average scores...")

        weights_filename = glob.glob(
            f"{tourney_folder}/multi_year_regression_weights.csv")

        # Load the weights CSV
        try:
            print(f"Attempting to read in {weights_filename}...")
            # Assume only 1 weights.csv
            weights_df = pd.read_csv(weights_filename[0])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Weights file {weights_filename} not found. Please run the model in training mode first.")

        print(f"\nExtracted regression weights from {weights_filename}:")
        print(weights_df)

        # Extract coefficients and intercept
        intercept = weights_df[weights_df['Feature']
                               # Finds Intercept row and gets coefficient value
                               == 'Intercept']['Coefficient'].iloc[0]
        coefficients = weights_df[weights_df['Feature'] != 'Intercept'].set_index(
            'Feature')['Coefficient'].to_dict()

        # Ensure all expected features are in the weights
        missing_features = [
            col for col in feature_cols if col not in coefficients]
        if missing_features:
            raise ValueError(
                f"Missing weights for features: {missing_features}. Ensure the weights CSV matches the current feature set.")

        # Prepare X (features) for prediction
        X = model_df[feature_cols]

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

    # Add the predictions as the second column (position 1, after 'PLAYERS')
    model_df.insert(loc=1, column='Predicted_Avg_Score',
                    value=rounded_predictions)

    print("\nPredicted_Avg_Score added to model_df.")

    return model_df


def calculate_win_probabilities(model_df):
    print("\nCalculating win probabilities from predicted average scores...")

    # Use Predicted_Avg_Score to calculate win probabilities
    beta = 0.5  # Scaling factor (adjust as needed)
    score_weights = np.exp(-beta * 4 * model_df['Predicted_Avg_Score'])

    # Normalize to get probabilities (sum to 1)
    total_weight = score_weights.sum()
    if total_weight == 0:
        print("Total weight is 0. Setting probabilities to 0.")
        probabilities = np.zeros(len(model_df))
    else:
        probabilities = score_weights / total_weight

    # Convert to percentages and round to 2 decimal places
    win_probabilities = np.round(probabilities * 100, 2)

    # Add the win_probabilities as the third column (position 2, after 'Predicted_Avg_Score')
    model_df.insert(loc=2, column='Win_Probability_%', value=win_probabilities)

    print("\nWin probabilities predicted.")

    print("\nmodel_df before reordering by win probability:")
    print(model_df)

    # Sort the DataFrame by Win_Probability_% in descending order
    model_df = model_df.sort_values(by='Win_Probability_%', ascending=False)


    return model_df


def main():
    parser = argparse.ArgumentParser(description="PGA Tour Model")
    parser.add_argument("model_year", type=int)
    parser.add_argument('--tourney-completed', action='store_true')

    args = parser.parse_args()

    tourney_folder = "CJ_Cup"

    try:
        # First get data from the current year's (model_year) tournament
        tourney, players, avg_scoring_to_model, odds_perc = load_current_scoring_csv_data(
            args.model_year, args.tourney_completed, tourney_folder)
        # Returns current year's player's normalized preivous average round scores tables (-0.1 if not played)
        past_scores_norm_df = load_all_prev_avg_scoring_normalized(
            args.model_year, players, tourney_folder)
        # Calculates weighted average of normalized past tournament z-scores
        model_df = compute_weighted_avg_scores(
            past_scores_norm_df, args.model_year, odds_perc, args.tourney_completed)
        # Merges Weighted_Avg_Past_Score_Z, (Odds_Probability_% or AVG_SCORE), and YTD normalized data
        model_df = load_ytd_data_and_normalize(
            model_df, args.model_year, tourney_folder)
        # Perform linear regression or predict using weights, based on tourney_completed flag
        model_df = perform_linear_regression(avg_scoring_to_model,
                                             model_df, args.model_year, tourney, args.tourney_completed, tourney_folder)
        # Calculate win probabilities
        model_df = calculate_win_probabilities(model_df)

        model_df = model_df.sort_values(
            by="Win_Probability_%", ascending=False)

        # Save to CSV
        output_file = f"{tourney_folder}/{tourney}_{args.model_year}_model.csv".replace(
            " ", "_")
        model_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
