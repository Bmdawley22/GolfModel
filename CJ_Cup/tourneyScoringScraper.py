import sys
import argparse
import pandas as pd
import glob
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Function to setup the web driver and return it


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    # chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Function to get the PGA schedule for a given year and return the list of tourney names in order of occurrence


def get_schedule(driver, year):
    print(
        f"\nRunning get_schedule on URL: https://www.pgatour.com/schedule/{year}...\n")
    try:
        driver.get(f"https://www.pgatour.com/schedule/{year}")

        # Wait until all <p> tags with class 'css-vgdvwe' are present
        all_tourneys_listed = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, "p.css-vgdvwe"))
        )

        schedule = [tourney.text for tourney in all_tourneys_listed]

        # Strip space at end of each string
        i = 0
        for tourney in schedule:
            schedule[i] = schedule[i][:-1]
            i += 1

        if schedule:
            print("Schedule extracted.")

        return schedule

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Function to promopt user to select tourney to model based on input schedule array and return the selected tourney


def select_tourney(tournaments):
    print(f"\nRunning select_tourney function...\n")

    if len(tournaments) > 0:
        for i, tournament in enumerate(tournaments, 1):
            print(f"{i}. {tournament}")

        # Prompt the user to select a tournament
        choice = 0
        while True:
            try:
                choice = int(
                    input("\nEnter the number of the tournament you want to model.\n "))
                if 1 <= choice <= len(tournaments):
                    # the list of tournaments to select starts at one so the index of the array is choice - 1
                    selected_tournament = tournaments[choice - 1]
                    break
                else:
                    print(
                        f"\nPlease enter a number between 1 and {len(tournaments)}.\n")
            except ValueError:
                print("\nPlease enter a valid number.\n")

    print(f"\nSelected Tournament: {selected_tournament}")

    return selected_tournament

# Function to extract tournament scoring data and write to csv


def get_tourney_scoring(driver, year, tourney, schedule):
    print(f"\nRunning get_tourney_scoring for {year} {tourney}...\n")
    try:
        driver.get(f"https://www.pgatour.com/schedule/{year}")

        print(f"Navigating to https://www.pgatour.com/schedule/{year}")

        # Get the parent link tag from the p tag matching tourney
        tourney_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//p[text()='{tourney}']"))
        ).find_element(By.XPATH, "./ancestor::a")

        driver.execute_script("arguments[0].click();", tourney_link)
        print(f"{tourney} link found and clicked")

        # Wait to make sure we've navigated to the correct tourney
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, f"//h1[contains(text(), '{tourney}')]"))
        )
        print(f"\nNavigated to {tourney}\n")

        # Step 1: Extract Headers
        header_cells = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.TAG_NAME, "thead"))
        ).find_elements(By.TAG_NAME, "th")

        print("Found scoring data table headers. Extracting data...")

        # Extract text from headers
        headers = []
        for cell in header_cells:
            # Some headers might have a <button> inside
            try:
                button = cell.find_element(By.TAG_NAME, "button")
                headers.append(button.text.strip())
            except:
                headers.append(cell.text.strip())

        print("Headers extracted.")

        # Find all rows in the table once table is loaded
        rows = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.TAG_NAME, "tbody"))
        ).find_elements(By.CSS_SELECTOR, "tr[class*='player-']")

        print("Found scoring data table rows. Extracting data...")

        # Prepare list for data
        data = []

        for row in rows:
            # Find all cells (td) in the row
            cells = row.find_elements(By.TAG_NAME, "td")
            cell_texts = [cell.text.strip() for cell in cells]
            if len(cell_texts) > 14:
                cell_texts = cell_texts[:-1]
            # If you want to get rid of the T (tie) in the position
            if "T" in cell_texts[1]:
                cell_texts[0].strip("T", "")
            # Only add if row has data
            if cell_texts and cell_texts[0] != "WD":
                data.append(cell_texts)

        print("\nScoring table extracted.\n")

        # Step 3: Create DataFrame
        df = pd.DataFrame(data, columns=headers[:len(data[0])])

        # # Remove last 3 columns
        df = df.iloc[:, :-3]

        # # Now drop "Round"
        df = df.drop(
            columns=[col for col in df.columns if col == "" or col == "ROUND"])

        print("Cleaned tabled columns.")

        # Make sure the numeric columns are actually numbers
        df["R1"] = pd.to_numeric(df["R1"], errors="coerce")
        df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
        df["STROKES"] = pd.to_numeric(df["STROKES"], errors="coerce")

        # Create Average Score column (score per round)
        df["AVG_SCORE"] = df.apply(
            lambda row: (row["R1"] + row["R2"]) /
            2 if row["POS"] == "CUT" else row["STROKES"] / 4,
            axis=1
        )
        print("Calcualted AVG_SCORE and added the column")

        df["AVG_SCORE_Z"] = (df["AVG_SCORE"].mean() -
                             df["AVG_SCORE"]) / df["AVG_SCORE"].std()

        print("Calculated normalized average score.")

        n_rows = len(df)

        # First rows with data, the rest blank
        year_col = [year] + [""] * (n_rows - 1)
        tourney_col = [tourney] + [""] * (n_rows - 1)
        schedule_col = schedule + [""] * (n_rows - len(schedule))

        # Add data information columns YEAR, TOURNAMENT, SCHEDULE
        df["YEAR"] = year_col
        df["TOURNAMENT"] = tourney_col
        df["SCHEDULE"] = schedule_col
        print("Added year, tourney, and schedule data to df")

        print("\nUpdated DataFrame:")
        print(df)

        # # Save to CSV
        # tourney = "cj_cup_byron_nelson"  # Set your tourney name dynamically if needed
        df.to_csv(
            f"{tourney.replace(" ", "_")}_{year}_scoring.csv", index=False)

    except Exception as e:
        print(f"Error occurred: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="PGA Tour Model")
    parser.add_argument("year")

    args = parser.parse_args()

    try:
        driver = setup_driver()

        schedule = get_schedule(driver, args.year)

        # commented out while developing
        tourney = select_tourney(schedule)

        # players, scores_per_rd =
        # hard coded tourney to be CJ cup for developing, "tourney" should be the arg
        get_tourney_scoring(driver, args.year, tourney, schedule)

    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
