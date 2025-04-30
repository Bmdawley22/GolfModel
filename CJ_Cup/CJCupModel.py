import sys
import argparse
import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Function to get the PGA schedule for a given year and return the list of tourney names in order of occurrence


def load_current_scoring_csv_data(year):
    # Find CSV file that ends with f"{year}_scoring.csv"
    files = glob.glob(f"*{year}_scoring.csv")
    if not files:
        raise FileNotFoundError(f"No Scoring CSV file found for year {year}.")

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
                    # result_df[str(year)] = -0.1
                    continue

                # Create a Series mapping players to their AVG_SCORE_Z values
                avg_score_z_dict = df.set_index(
                    'PLAYER')['AVG_SCORE_Z'].to_dict()

                # For each player, get their AVG_SCORE_Z or set to -0.1 if not found (0.1 stdev penalty for not having played tourney)
                avg_score_z = []
                for player in players:
                    if player in avg_score_z_dict:
                        # Ensure float
                        avg_score_z.append(
                            round(float(avg_score_z_dict[player]), 4))
                    else:
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


def compute_weighted_avg_scores(avg_scoring_to_model, past_scores_df, model_year):
    # Get the list of years (columns except 'PLAYERS')
    years = [int(col) for col in past_scores_df.columns if col != 'PLAYERS']

    print(f"Years in table: {years}")

    if not years:
        print("No past scoring data available. Setting weighted average to 0.")
        return pd.Series(0, index=past_scores_df.index)

    # Compute weights using exponential decay
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
    past_scores_df['Weighted_Avg_Past_Score_Z'] = np.round(weighted_avg, 2)

    print("\nUpdated past_scores_df with weighted averages:\n")
    print(past_scores_df)

    # Remove all year columns (keep only 'PLAYERS' and 'Weighted_Avg_Past_Score_Z')
    year_columns = [col for col in past_scores_df.columns if col not in [
        'PLAYERS', 'Weighted_Avg_Past_Score_Z']]
    past_scores_df = past_scores_df.drop(columns=year_columns)

    # Add a new column between 'PLAYERS' and 'Weighted_Avg_Past_Score_Z'
    # For this example, let's call the new column 'New_Column' and fill it with a placeholder value (e.g., 0)
    # You can replace 0 with the actual data for the new column
    past_scores_df.insert(
        loc=1, column=f'{model_year}_Avg_Score', value=avg_scoring_to_model)

    # Display the updated DataFrame
    print("\nUpdated past_scores_df after removing year columns and adding new column:")
    print(past_scores_df)

    return past_scores_df


def load_ytd_data_and_normalize(model_df, model_year):
    # Find CSV file that ends with f"ytd_thru*{year}.csv"
    files = glob.glob(f"ytd_thru*{model_year}.csv")
    if not files:
        raise FileNotFoundError(
            f"No YTD CSV file found for year {model_year}.")

    csv_file = files[0]  # Assume only one match
    print(f"Loading data from {csv_file}...")

    # Read the CSV
    df = pd.read_csv(csv_file)

    print(f"Data loaded from {csv_file}...")
    print(df.head)

    print("Normalizing data.")

    # Identify numeric columns (all except 'PLAYERS')
    numeric_cols = df.columns.drop('PLAYERS')

    # Columns to invert normalization (adjust as needed for your data)
    invert_cols = ['Distance From Edge of Fairway', "Rough Tendency", "Fairway Proximity", "Approaches from > 200", "Proximity from Sand (Short)",
                   "Proximity from Rough (Short)", "Proximity from 30+", "Proximity ATG", "3-Putt Avoidance"]

    # Apply Z-score normalization
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

    merged_df = model_df.merge(df, on='PLAYERS', how='inner')

    return merged_df


def perform_linear_regression(model_df, model_year, tourney):
    print("\nPerforming linear regression to predict average score...")

    # Define the target variable (y) and features (X)
    target_col = f"{model_year}_Avg_Score"
    feature_cols = [col for col in model_df.columns if col not in [
        'PLAYERS', target_col]]

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

    # Add the predictions as the second column (position 1, after 'PLAYERS')
    model_df.insert(loc=1, column='Predicted_Avg_Score',
                    value=rounded_predictions)

    # # Print the coefficients for debugging
    # print("\nLinear Regression Coefficients:")
    # for feature, coef in zip(feature_cols, model.coef_):
    #     print(f"{feature}: {coef:.4f}")
    # print(f"Intercept: {model.intercept_:.4f}")

    # Create a DataFrame with the coefficients and intercept
    weights_df = pd.DataFrame({
        'Feature': feature_cols + ['Intercept'],
        'Coefficient': list(model.coef_) + [model.intercept_]
    })

    # Round coefficients to 4 decimal places for readability
    weights_df['Coefficient'] = weights_df['Coefficient'].round(4)

    # Save the coefficients to a CSV file
    weights_filename = f"{tourney}_{model_year}_weights.csv".replace(
        " ", "_")
    weights_df.to_csv(weights_filename, index=False)
    print(f"Regression weights saved to {weights_filename}")

    return model_df


def calculate_win_probabilities(model_df):
    print("\nCalculating win probabilities from predicted average scores...")

    # Use Predicted_Avg_Score to calculate win probabilities
    beta = 0.5  # Scaling factor (adjust as needed)
    score_weights = np.exp(-beta * 4*model_df['Predicted_Avg_Score'])

    # Normalize to get probabilities (sum to 1)
    total_weight = score_weights.sum()
    if total_weight == 0:
        print("Total weight is 0. Setting probabilities to 0.")
        probabilities = np.zeros(len(model_df))
    else:
        probabilities = score_weights / total_weight

    # Convert to percentages and round to 2 decimal places
    win_probabilities = np.round(probabilities * 100, 2)

    # Add the win_probabilities as the second column (position 2, after 'PLAYERS')
    model_df.insert(loc=2, column='Win_Probability_%',
                    value=win_probabilities)

    return model_df


def main():
    parser = argparse.ArgumentParser(description="PGA Tour Model")
    parser.add_argument("model_year")

    args = parser.parse_args()

    try:
        tourney, players, avg_scoring_to_model = load_current_scoring_csv_data(
            args.model_year)

        past_scores_norm_df = load_all_prev_avg_scoring_normalized(
            args.model_year, players)

        model_df = compute_weighted_avg_scores(avg_scoring_to_model,
                                               past_scores_norm_df, args.model_year)

        model_df = load_ytd_data_and_normalize(model_df, args.model_year)

        # Perform linear regression and add predictions
        model_df = perform_linear_regression(
            model_df, args.model_year, tourney)

        # Calculate win probabilities
        model_df = calculate_win_probabilities(model_df)

        # Save to CSV
        output_file = f"{tourney}_{args.model_year}_model.csv".replace(
            " ", "_")
        model_df.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
