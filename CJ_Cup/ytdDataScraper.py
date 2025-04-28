import argparse
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# Set up Selenium with Chrome to avoid bot detection


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument(
        "--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--headless")  # Run in headless mode
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


def select_tourney(driver, prev_selected_tourney='', prev_tourney_before=''):
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

        # Only want to select the tourney once
        if len(prev_selected_tourney) == 0:
            for i, tournament in enumerate(Tournaments, 1):
                print(f"{i}. {tournament}")

            # Prompt the user to select a tournament
            choice = 0
            while True:
                try:
                    choice = int(
                        input("\nEnter the number of the tournament you are modeling.\nNOTE: Select the tournament before the one you are modeling:\n "))
                    if 1 <= choice <= len(Tournaments):
                        # the list of tournaments to select starts at one so the index of the array is choice - 1
                        selected_tournament = Tournaments[choice - 1]
                        # save tournament before in case the next stat doesn't have this tourney
                        tourney_before_selected = Tournaments[choice]
                        break
                    else:
                        print(
                            f"\nPlease enter a number between 1 and {len(Tournaments)}.\n")
                except ValueError:
                    print("\nPlease enter a valid number.\n")
        else:
            # set tourney_before_selected variable as we wnat to keep this info through each run
            tourney_before_selected = prev_tourney_before
            # check if selected tourney button is available for the current stat we're running
            if prev_selected_tourney in Tournaments:
                selected_tournament = prev_selected_tourney
            else:
                # if prev_selected_tourney not shown for current stat, move to the tourney before
                print(
                    f"{prev_selected_tourney} not found. Moving to tournament before: {tourney_before_selected}")
                selected_tournament = tourney_before_selected

        print(f"\Tournament to select: {selected_tournament}")

        # Find the button with text the same as the selected tournament
        selected_button = driver.find_element(
            By.XPATH, f"//button[text()='{selected_tournament}']")
        print(f"{selected_tournament} button found.")

        driver.execute_script(
            "arguments[0].click();", selected_button)
        print(f"{selected_tournament} button clicked.\n")

        time.sleep(2)

        if len(prev_selected_tourney) > 0:
            selected_tournament = prev_selected_tourney

        return selected_tournament, tourney_before_selected

    except Exception as e:
        print(f"Error occurred with selecting tournament: {e}")
        print(Tournaments)
        driver.quit()
        sys.exit()


def extract_table(driver, stat_name):
    print(f"Extracting {stat_name} data...\n")
    # Find all rows in the table
    rows = driver.find_elements(By.CLASS_NAME, "css-paaamq")

    # Lists to store data
    players = []
    averages = []

    # Skip the header row (row-0) and process data rows
    for row in rows:
        # Extract cells in the row
        cells = row.find_elements(By.TAG_NAME, "span")

        # Check if the row has the expected number of cells (5 for data rows)
        if len(cells) < 6:
            continue

        try:
            distance_stat_names = ["Distance From Edge of Fairway", "Fairway Proximity", "Approaches from > 200",
                                   "Rough Proximity", "Approaches from Rough > 200", "Approaches from Rough > 200",
                                   "Proximity from Sand (Short)", "Proximity from Rough (Short)", "Proximity from 30+",
                                   "Proximity ATG"]
            percent_stat_names = ["Rough Tendency", "GIR", "Going for Green", "GIR from Other than Fairway",
                                  "3-Putt Avoidance", "Putting 5-15ft"]

            if stat_name in distance_stat_names:
                if stat_name in ["Approaches from > 200", "Approaches from Rough > 200", "Proximity from Sand (Short)", "Proximity from Rough (Short)", "Proximity from 30+", "Proximity ATG"]:
                    # Player name is in the 4th cell from the end
                    player = cells[-5].text.strip()

                    # Calculate average using Total Feet divided by Total Strokes
                    total_feet = float(cells[-3].text.strip().replace(",", ""))
                    total_strokes = float(
                        cells[-2].text.strip().replace(",", ""))
                    avg = round(total_feet / total_strokes, 2)
                elif stat_name == "Fairway Proximity":
                    # Player name is in the 4th cell from the end
                    player = cells[-5].text.strip()

                    # Calculate average using Total Feet divided by Total Strokes
                    total_feet = float(cells[-2].text.strip().replace(",", ""))
                    total_strokes = float(
                        cells[-3].text.strip().replace(",", ""))
                    avg = round(total_feet / total_strokes, 2)
                else:
                    # Player name is in the 4th cell from the end
                    player = cells[-4].text.strip()

                    # Calculate average using Total Feet divided by Total Strokes
                    total_feet = float(cells[-2].text.strip().replace(",", ""))
                    total_strokes = float(
                        cells[-1].text.strip().replace(",", ""))
                    avg = round(total_feet / total_strokes, 2)
            elif stat_name in percent_stat_names:
                if stat_name in ["GIR from Other than Fairway", "3-Putt Avoidance", "Putting 5-15ft"]:
                    # Player name is in the 5th cell from the end
                    player = cells[-4].text.strip()
                    # Avg data is in the 4th cell from the end
                    avg = round(float(cells[-3].text.strip('%')) / 100, 4)
                elif stat_name == "Going for Green":
                    # Player name is in the 5th cell from the end
                    player = cells[-6].text.strip()
                    # Avg data is in the 4th cell from the end
                    avg = round(float(cells[-5].text.strip('%')) / 100, 4)
                else:
                    # Player name is in the 5th cell from the end
                    player = cells[-5].text.strip()
                    # Avg data is in the 4th cell from the end
                    avg = round(float(cells[-4].text.strip('%')) / 100, 4)
            else:
                # Player name is in the 4th cell from the end
                player = cells[-4].text.strip()
                # Avg data is in the 3rd cell from the end
                avg = cells[-3].text.strip()

            players.append(player)
            averages.append(float(avg))
        except Exception as e:
            print(
                f"Error processing row for player {player if 'player' in locals() else 'unknown'}: {e}")
            continue

    print(f"////////////\n{stat_name} data extracted.\n////////////")

    return players, averages


def scrape_pga_table(url, stat_name="Average_Driving_Distance", stat_year="2024", selected_tourney='', tourney_before_selected=''):
    driver = setup_driver()
    try:
        # Navigate to the page
        driver.get(url)

        # Wait for the table to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "chakra-table"))
        )

        # Add a small delay to ensure all rows are loaded
        time.sleep(2)

        select_year(driver, stat_year)

        time.sleep(2)

        selected_tourney, tourney_before_selected = select_tourney(
            driver, selected_tourney, tourney_before_selected)

        players, averages = extract_table(driver, stat_name)

        # Create a DataFrame
        if players and averages:
            data = {"Player": players, stat_name: averages}
            df = pd.DataFrame(data)
            return df, selected_tourney, tourney_before_selected
        else:
            print("No data extracted.")
            return None, selected_tourney, tourney_before_selected

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, selected_tourney, tourney_before_selected

    finally:
        driver.quit()

# Main function


def main():
    parser = argparse.ArgumentParser(
        description="Download and rename FanGraphs CSVs based on date range.")
    parser.add_argument("year")

    args = parser.parse_args()

    # List of URLs and corresponding stat names
    # You'll update this list with the actual URLs and stat names
    url_list = [
        {"url": "https://www.pgatour.com/stats/detail/101",
            "stat_name": "Average Driving Distance"},
        {"url": "https://www.pgatour.com/stats/detail/02402",
            "stat_name": "Ball Speed"},
        {"url": "https://www.pgatour.com/stats/detail/02420",
            "stat_name": "Distance From Edge of Fairway"},
        {"url": "https://www.pgatour.com/stats/detail/02435",
            "stat_name": "Rough Tendency"},
        {"url": "https://www.pgatour.com/stats/detail/103",
            "stat_name": "GIR"},
        {"url": "https://www.pgatour.com/stats/detail/431",
            "stat_name": "Fairway Proximity"},
        {"url": "https://www.pgatour.com/stats/detail/336",
            "stat_name": "Approaches from > 200"},
        {"url": "https://www.pgatour.com/stats/detail/419",
            "stat_name": "Going for Green"},
        {"url": "https://www.pgatour.com/stats/detail/199",
            "stat_name": "GIR from Other than Fairway"},
        {"url": "https://www.pgatour.com/stats/detail/375",
            "stat_name": "Proximity from Sand (Short)"},
        {"url": "https://www.pgatour.com/stats/detail/376",
            "stat_name": "Proximity from Rough (Short)"},
        {"url": "https://www.pgatour.com/stats/detail/379",
            "stat_name": "Proximity from 30+"},
        {"url": "https://www.pgatour.com/stats/detail/374",
            "stat_name": "Proximity ATG"},
        {"url": "https://www.pgatour.com/stats/detail/426",
            "stat_name": "3-Putt Avoidance"},
        {"url": "https://www.pgatour.com/stats/detail/02327",
            "stat_name": "Putting 5-15ft"}
    ]
    # List to store DataFrames
    all_dfs = []
    selected_tourney = ''
    tourney_before_selected = ''

    # Loop through each URL and scrape the data
    for url_info in url_list:
        url = url_info["url"]
        stat_name = url_info["stat_name"]
        print(
            f"\nScraping data from {url} for {stat_name} for year {args.year}...")
        df, selected_tourney, tourney_before_selected = scrape_pga_table(
            url, stat_name, args.year, selected_tourney, tourney_before_selected)
        if df is not None:
            all_dfs.append(df)
        else:
            print(f"Failed to scrape data from {url}")

    # Combine all DataFrames on the "Player" column
    if all_dfs:
        # Start with the first DataFrame
        combined_df = all_dfs[0]

        # Merge with the remaining DataFrames
        for df in all_dfs[1:]:
            combined_df = combined_df.merge(df, on="Player", how="outer")

        # Sort by Player name for consistency
        combined_df = combined_df.sort_values("Player").reset_index(drop=True)

        # Print the combined DataFrame
        print("\nCombined DataFrame:")
        print(combined_df)

        # # Save to CSV
        combined_df.to_csv(
            f"ytd_thru_{selected_tourney}_{args.year}.csv", index=False)
        print(f"Data exported to ytd_thru_{selected_tourney}_{args.year}.csv")
        # combined_df.to_csv(
        #     f"temp.csv", index=False)
        # print(f"Data exported to temp.csv")

        # Export Stat Name URLS to stat_name_urls.csv
        # df = pd.DataFrame(url_list)[["stat_name", "url"]]
        # df.columns = ["Stat Names", "URLs"]

        # df.to_csv(f"stat_names_urls.csv", index=False)
    else:
        print("No data to combine.")


if __name__ == "__main__":
    main()
