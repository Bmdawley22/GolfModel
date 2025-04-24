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
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Scrape the table from the PGA Tour stats page


def select_year(driver, year_to_select):
    print("Running select_year function")

    try:
        # Wait for the dropdown button to be clickable (button with a child span containing exactly "Season")
        dropdown = WebDriverWait(driver, 15).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'button[aria-label="Season"]')))

        print("Season dropdown button found.")

        driver.execute_script("arguments[0].click();", dropdown)

        print("Season dropdown clicked")

        # Find the year option using its text
        year_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//div[contains(@id, 'menu-list-')]//button[contains(text(), '{year_to_select}')]"))
        )

        print(f"{year_to_select} option found")

        driver.execute_script("arguments[0].click();", year_option)

        print("Year clicked.")

        time.sleep(2)

        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//p[contains(text(), '{year_to_select}')]"))
        )

        print(f"Year changed to: {year_to_select}")

    except Exception as e:
        print(f"Error occurred with selecting year: {e}")
        driver.quit()
        sys.exit()


def select_tourney(driver, prev_selected_tourney=''):
    print("Running select_tourney function")

    try:
        # Wait for all elements with id containing "menu-button" to be present
        menu_buttons = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located(
                (By.XPATH, "//*[contains(@id, 'menu-button')]"))
        )

        # print(len(menu_buttons))

        i = 1
        for menu_button in menu_buttons:

            span_elements = menu_button.find_elements(
                By.XPATH, ".//span[text()='Tournament']")

            if span_elements:
                # print(
                #     f"A child span of menu button {i} with the text 'Tournament' was found.")

                driver.execute_script(
                    "arguments[0].click();", menu_button)
                print("Tournament dropdown clicked")
            # else:
                # print(
                # f"No child span of menu buttons {i} found with the text 'Tournament' was found.")
            i += 1

        # Wait for the button with the text "The Sentry" to be located
        sentry_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located(
                (By.XPATH, "//button[text()='The Sentry']"))
        )
        # print("The Sentry Tournament button found.")

        # Find the parent div of the Masters Tournament button
        tournament_menu = sentry_button.find_element(
            By.XPATH, "./ancestor::div[contains(@class, 'chakra-menu__menu-list')]")
        print("Parent tournament dropdown menu found.")

        # Find all child buttons within the parent div
        tournament_buttons = tournament_menu.find_elements(
            By.TAG_NAME, "button")

        # Extract the text of each button into the Tournaments array
        Tournaments = [button.text.strip() for button in tournament_buttons]

        for i, tournament in enumerate(Tournaments, 1):
            print(f"{i}. {tournament}")

        if len(prev_selected_tourney) == 0:
            # Prompt the user to select a tournament
            choice = 0
            while True:
                try:
                    choice = int(
                        input("Enter the number of the tournament you want YTD data for.  NOTE: the tournament before the tournament you choose will be selected as we want YTD stats UP TO that tournament: "))
                    if 1 <= choice <= len(Tournaments):
                        # the list of tournaments to select starts at one so the index of the array is choice - 1
                        selected_tournament = Tournaments[choice - 1]
                        # Tournament before is lower in the list (index = selected_tournament index + 1)
                        ytd_tourney = Tournaments[choice]
                        break
                    else:
                        print(
                            f"Please enter a number between 1 and {len(Tournaments)}.")
                except ValueError:
                    print("Please enter a valid number.")
        else:
            selected_tournament = prev_selected_tourney
            ytd_tourney = Tournaments[Tournaments.index(
                prev_selected_tourney) + 1]

        print(f"Selected Tournament: {selected_tournament}")

        # Find the button with the text "Masters Tournament"
        ytd_tourney_button = driver.find_element(
            By.XPATH, f"//button[text()='{ytd_tourney}']")
        print(f"{ytd_tourney} button found.")

        driver.execute_script(
            "arguments[0].click();", ytd_tourney_button)
        print(f"{ytd_tourney} button clicked.")

        time.sleep(5)

        return selected_tournament

        # driver.quit()
        # sys.exit()

    except Exception as e:
        print(f"Error occurred with selecting tournament: {e}")
        driver.quit()
        sys.exit()


def scrape_pga_table(url, stat_name="Average_Driving_Distance", stat_year="2024", selected_tourney=''):
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

        selected_tourney = select_tourney(driver, selected_tourney)

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
                print(
                    f"Skipping row with {len(cells)} cells: {[cell.text for cell in cells]}")
                continue
            # else:
                # print(
                #     f"Save row with {len(cells)} cells: {[cell.text for cell in cells]}")

            try:
                # Player name is in the 4th cell from the end
                player = cells[-4].text.strip()

                if stat_name == "Distance_From_Edge_of_Fairway":
                    # Calculate average using Total Feet divided by Total Strokes
                    total_feet = float(cells[-2].text.strip().replace(",", ""))
                    total_strokes = float(
                        cells[-1].text.strip().replace(",", ""))
                    avg = round(total_feet / total_strokes, 2)
                else:
                    # Player name is in the 4th cell from the end
                    avg = cells[-3].text.strip()

                players.append(player)
                averages.append(float(avg))
            except Exception as e:
                print(
                    f"Error processing row for player {player if 'player' in locals() else 'unknown'}: {e}")
                continue

        print(f"////////////\n{stat_name} data extracted.\n////////////")

        # Create a DataFrame
        if players and averages:
            data = {"Player": players, stat_name: averages}
            df = pd.DataFrame(data)
            return df, selected_tourney
        else:
            print("No data extracted.")
            return None, selected_tourney

    except Exception as e:
        print(f"Error occurred: {e}")
        return None, selected_tourney

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
            "stat_name": "Average_Driving_Distance"},
        {"url": "https://www.pgatour.com/stats/detail/02402",
            "stat_name": "Ball_Speed"},
        {"url": "https://www.pgatour.com/stats/detail/02420",
            "stat_name": "Distance_From_Edge_of_Fairway"}
    ]

    # List to store DataFrames
    all_dfs = []
    selected_tourney = ''

    # Loop through each URL and scrape the data
    for url_info in url_list:
        url = url_info["url"]
        stat_name = url_info["stat_name"]
        print(
            f"Scraping data from {url} for {stat_name} for year {args.year}...")
        df, selected_tourney = scrape_pga_table(
            url, stat_name, args.year, selected_tourney)
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

        # Save to CSV
        combined_df.to_csv("pga_stats_combined.csv", index=False)
        print("Data saved to pga_stats_combined.csv")
    else:
        print("No data to combine.")


if __name__ == "__main__":
    main()
