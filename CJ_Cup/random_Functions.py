import sys
import pandas as pd
import glob
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time



def find_csv_file():
    # Search for CSV files with "ytd_thru_" in the name
    files = glob.glob("*ytd_thru_*.csv")
    if not files:
        raise FileNotFoundError(
            "No CSV file found with 'ytd_thru_' in the name.")
    if len(files) > 1:
        print(f"Multiple matching files found: {files}")
        print(f"Using the first file: {files[0]}")
    return files[0]

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    # chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def select_year(driver, year_to_select):
    print("\nRunning select_year function...")
    try:
        # Wait for the dropdown button to be clickable (button with a child span containing exactly "Season")
        dropdown = WebDriverWait(driver, 15).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'button[aria-label="Season"]')))
        driver.execute_script("arguments[0].click();", dropdown)
        print("Season dropdown found and clicked")

        # Find the button corresponding to the year_to_select
        year_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//div[contains(@id, 'menu-list-')]//button[contains(text(), '{year_to_select}')]"))
        )
        driver.execute_script("arguments[0].click();", year_option)
        print(f"{year_to_select} option found and clicked")

        # Wait for year to change
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//p[contains(text(), '{year_to_select}')]"))
        )
        print(f"Year changed to: {year_to_select}\n")

    except Exception as e:
        print(f"\nError occurred with selecting year: {e}\n")
        driver.quit()
        sys.exit()

def select_tourney(driver, ytd_tourney):
    print("Running select_tourney function...")
    try:
        # Wait for all elements with id containing "menu-button" to be present
        menu_buttons = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//*[contains(@id, 'menu-button')]"))
        )

        i = 1
        for menu_button in menu_buttons:
            span_elements = menu_button.find_elements(
                By.XPATH, ".//span[text()='Tournament']")

            if span_elements:
                driver.execute_script(
                    "arguments[0].click();", menu_button)
                print("Tournament dropdown clicked")
            i += 1

        # Wait for the button with the text "Masters Tournament" to be located
        # Note: won't be able to locate if you're modeling a tourney before the masters
        tourney_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[text()='Masters Tournament' or text()='The Sentry']"))
        )

        print("Masters or Sentry button found")

        # Find the parent div of the Masters Tournament button
        tournament_menu = tourney_button.find_element(
            By.XPATH, "./ancestor::div[contains(@class, 'chakra-menu__menu-list')]")
        print("Tournament dropdown menu found.")

        # Find all child buttons within the parent div
        tournament_buttons = tournament_menu.find_elements(
            By.TAG_NAME, "button")

        # Extract the text of each button into the Tournaments array
        Tournaments = [button.text.strip() for button in tournament_buttons]

        tourney_to_select = Tournaments[Tournaments.index(ytd_tourney) - 1]

        print(f"\nTournament to select: {tourney_to_select}")

        # Find the button with text the same as the selected tournament
        selected_button = driver.find_element(
            By.XPATH, f"//button[text()='{tourney_to_select}']")

        driver.execute_script(
            "arguments[0].click();", selected_button)
        print(f"{tourney_to_select} button found and  clicked.\n")

        return tourney_to_select

    except Exception as e:
        print(f"Error occurred with selecting tournament: {e}")
        driver.quit()
        sys.exit()

def select_tournament_only(driver, tourney_name):
    print("\nRunning select_tournament_only function...")
    try:
        # Wait for the dropdown button to be clickable (button with a child span containing exactly "Season")
        dropdown = WebDriverWait(driver, 15).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'button[aria-label="Time Period"]')))
        driver.execute_script("arguments[0].click();", dropdown)
        print("Time Period dropdown found and clicked")

        # Find the button corresponding to the year_to_select
        tourney_only_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//div[contains(@id, 'menu-list-')]//button[contains(text(), 'Tournament Only')]"))
        )
        driver.execute_script("arguments[0].click();", tourney_only_option)
        print("Tournament Only option found and clicked")

        # Wait for year to change
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//p[contains(text(), '{tourney_name}')]"))
        )
        print(f"Time period changed to Tournament only\n")
    except Exception as e:
        print(f"\nError occurred with selecting year: {e}\n")
        driver.quit()
        sys.exit()

def extract_scoring_data(driver):
    try:
        print(f"Extracting scoring data...\n")
        # Find all rows in the table
        rows = driver.find_elements(By.CLASS_NAME, "css-paaamq")

        # Lists to store data
        players = []
        scores_per_rd = []

        # Skip the header row (row-0) and process data rows
        for row in rows:
            try:
                # Extract cells in the row
                cells = row.find_elements(By.TAG_NAME, "span")

                if len(cells) < 3:
                    continue

                player = cells[-5].text.strip()
                strokes = float(cells[-3].text.strip())
                rounds = float(cells[-1].text.strip())
                print(f"{strokes} / {rounds}")
                score_per_rd = round(float(strokes / (100 * rounds)))

                players.append(player)
                scores_per_rd.append(float(score_per_rd))

        
            except Exception as e:
                print(
                    f"Error processing row for player {player if 'player' in locals() else 'unknown'}: {e}")
                continue
        print(len(players))
        # print(players)
        print(len(scores_per_rd))
        # print(scores_per_rd)
        return players, scores_per_rd
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def getTourneyScoring(driver, year_to_select, ytd_tourney):
    try:
        driver.get("https://www.pgatour.com/stats/detail/02675")

        # Make sure "Scoring Average" page is loaded
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((
            By.XPATH, '//h1[contains(text(), "Scoring Average")]')))
        
        select_year(driver, year_to_select)
        time.sleep(2)

        tourney_name = select_tourney(driver, ytd_tourney)
        time.sleep(2)

        select_tournament_only(driver, tourney_name)
        time.sleep(2)

        extract_scoring_data(driver)

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

    finally:
        driver.quit()

def normalize_df(csv_file):
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(csv_file)

        # Select only numeric columns to normalize
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Columns to invert normalization (adjust as needed for your data)
        invert_cols = ["Rough Tendency", "Fairway Proximity", "Approaches from > 200", "Proximity from Sand (Short)", 
                       "Proximity from Rough (Short)", "Proximity from 30+", "Proximity ATG", "3-Putt Avoidance"]

        # Apply Z-score normalization
        for col in numeric_cols:
            if col in invert_cols:
                # Inverted normalization
                df[col] = -1 * (df[col] - df[col].mean()) / df[col].std()
                print(f"{col} inverted.")
            else:
                # Regular normalization
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        # Truncate all data to 4 decimals
        df = df.round(4)

        print("Normalized DataFrame preview:")
        print(df.head())

        if df is not None:
            # Save the normalized DataFrame to a new CSV
            output_file = csv_file.replace('.csv', '_normalized.csv')
            df.to_csv(output_file, index=False)
            print(f"Normalized data saved to {output_file}")

        return df

    except Exception as e:
        print(f"Error during normalization: {e}")
        return None

def main():
    try:
        # Find the CSV file
        csv_file = find_csv_file()
        print(f"Reading CSV file: {csv_file}")

        filename = os.path.basename(csv_file)
        # Remove 'ytd_thru_' prefix and '.csv' suffix
        core_name = filename.replace('ytd_thru_', '').replace('.csv', '').replace('normalized', '')
        # Split on the last underscore to separate tournament name and year
        ytd_tourney, year = core_name.rsplit('_', 1)

        print(f"Extracted Tournament Name: {ytd_tourney}")
        print(f"Extracted Year: {year}")

        driver = setup_driver()

        players, scores_per_rd = getTourneyScoring(driver, year, ytd_tourney)

        # Create a DataFrame
        if players and scores_per_rd:
            data = {"Player": players, "Score Per Round": scores_per_rd}
            tourney_scoring_df = pd.DataFrame(data)
            print("Tourney Scoring DataFrame preview:")
            print(tourney_scoring_df.head())

        # Normalize YTD Data
        # normalize_df(csv_file)

        

    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()
