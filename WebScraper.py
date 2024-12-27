from io import StringIO
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import List, Optional


class PremierLeagueScraper:
    BASE_URL = "https://fbref.com"

    def __init__(self, start_year: int, end_year: int):
        """
        Initialize the scraper with a range of years and a requests session.
        """
        self.years = list(range(start_year, end_year - 1, -1))
        self.session = requests.Session()  # Use a session for persistent settings across requests
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; PremierLeagueScraper/1.0)'
        })

    def get_team_urls(self, standings_url: str) -> tuple[List[str], str]:
        """
        Fetches team URLs and the link to the previous season's standings page.
        """
        response = self.session.get(standings_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first table with class 'stats_table' and extract team links
        standings_table = soup.select('table.stats_table')[0]
        links = [l.get("href") for l in standings_table.find_all('a')]
        team_links = [l for l in links if '/squads/' in l]
        team_urls = [f"{self.BASE_URL}{l}" for l in team_links]

        # Find the link to the previous season
        prev_season = soup.select("a.prev")[0].get("href")
        return team_urls, f"{self.BASE_URL}{prev_season}"

    def get_team_data(self, team_url: str, year: int) -> Optional[pd.DataFrame]:
        """
        Scrapes match and shooting data for a specific team in a given season.
        """
        try:
            # Extract the team name from the URL
            team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

            # Fetch the team's match data
            response = self.session.get(team_url)
            response.raise_for_status()
            matches = pd.read_html(StringIO(response.text), match="Scores & Fixtures")[0]  # Parse match data table

            # Parse the page to find the link to shooting stats
            soup = BeautifulSoup(response.text, 'html.parser')
            shooting_links = [l.get("href") for l in soup.find_all('a')
                              if l and 'all_comps/shooting/' in l.get("href", "")]

            if not shooting_links:
                return None

            # Fetch the shooting stats
            shooting_response = self.session.get(f"{self.BASE_URL}{shooting_links[0]}")
            shooting_response.raise_for_status()
            shooting = pd.read_html(StringIO(shooting_response.text), match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()  # Drop the multi-index header

            # Merge match data and shooting stats on the 'Date' column
            team_data = matches.merge(
                shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]],
                on="Date"
            )

            # Filter for Premier League matches only
            team_data = team_data[team_data["Comp"] == "Premier League"]
            team_data["Season"] = year
            team_data["Team"] = team_name

            return team_data

        except Exception as e:
            print(f"Error processing {team_url}: {str(e)}")
            return None

    def scrape(self) -> pd.DataFrame:
        """
        Main method to scrape data for all teams and seasons.
        """
        all_matches = []
        standings_url = f"{self.BASE_URL}/en/comps/9/stats/Premier-League-Stats"

        # Iterate through the years in descending order
        for year in self.years:
            print(f"Processing season {year}")
            team_urls, standings_url = self.get_team_urls(standings_url)  # Get team URLs for the current season

            for team_url in team_urls:
                team_data = self.get_team_data(team_url, year)
                if team_data is not None:
                    all_matches.append(team_data)
                time.sleep(1)

        # Combine all team data into a single DataFrame
        match_df = pd.concat(all_matches)
        match_df.columns = [c.lower() for c in match_df.columns]
        return match_df


if __name__ == "__main__":
    scraper = PremierLeagueScraper(2023, 2020)
    matches = scraper.scrape()
    matches.to_csv("matches.csv", index=False)
