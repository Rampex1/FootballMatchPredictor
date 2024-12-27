import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from typing import List, Tuple, Dict


class FootballPredictor:
    def __init__(self, data_path: str):
        """
        Initialize the football match prediction model.

        Args:
            data_path: Path to the CSV file containing match data
        """
        self.rf = RandomForestClassifier(
            n_estimators=50,
            min_samples_split=10,
            random_state=1
        )
        self.matches = self._load_and_clean_data(data_path)

        # Define columns for rolling averages
        self.stat_columns = [
            "gf",  # Goals For
            "ga",  # Goals Against
            "sh",  # Shots
            "sot",  # Shots on Target
            "dist",  # Average Shot Distance
            "fk",  # Free Kicks
            "pk",  # Penalties Scored
            "pkatt"  # Penalty Attempts
        ]

        self.team_mappings = {
            "Brighton and Hove Albion": "Brighton",
            "Manchester United": "Manchester Utd",
            "Newcastle United": "Newcastle Utd",
            "Tottenham Hotspur": "Tottenham",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves"
        }

    def _load_and_clean_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess the match data.

        Args:
            data_path: Path to the CSV file

        Returns:
            Cleaned DataFrame with encoded categorical variables
        """
        matches = pd.read_csv(data_path, index_col=0)

        # Convert and encode features
        matches["date"] = pd.to_datetime(matches["date"])
        matches["venue_code"] = matches["venue"].astype("category").cat.codes
        matches["opp_code"] = matches["opponent"].astype("category").cat.codes
        matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
        matches["day_code"] = matches["date"].dt.dayofweek

        # Create target variable (1 for win, 0 for draw/loss)
        matches["target"] = (matches["result"] == "W").astype("int")

        return matches

    def calculate_rolling_averages(self, group: pd.DataFrame, window: int = 3) -> pd.DataFrame:
        """
        Calculate rolling averages for specified statistics.

        Args:
            group: DataFrame containing team's matches
            window: Number of matches to use for rolling average

        Returns:
            DataFrame with added rolling average columns
        """
        group = group.sort_values("date")
        new_cols = [f"{col}_rolling" for col in self.stat_columns]

        rolling_stats = group[self.stat_columns].rolling(
            window,
            closed='left'
        ).mean()

        group[new_cols] = rolling_stats
        return group.dropna(subset=new_cols)

    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare feature set including rolling averages for all teams.

        Returns:
            DataFrame with all features prepared for modeling
        """
        # Calculate rolling averages for each team
        matches_rolling = self.matches.groupby("team").apply(
            lambda x: self.calculate_rolling_averages(x)
        )

        # Reset index for easier handling
        matches_rolling = matches_rolling.droplevel('team')
        matches_rolling.index = range(matches_rolling.shape[0])

        return matches_rolling

    def make_predictions(
            self,
            data: pd.DataFrame,
            cutoff_date: str = '2022-01-01'
    ) -> Tuple[pd.DataFrame, float]:
        """
        Train model and make predictions.

        Args:
            data: Prepared DataFrame with all features
            cutoff_date: Date to split train/test data

        Returns:
            Tuple containing predictions DataFrame and precision score
        """
        # Basic features plus rolling averages
        predictors = [
                         "venue_code",
                         "opp_code",
                         "hour",
                         "day_code"
                     ] + [f"{col}_rolling" for col in self.stat_columns]

        # Split data into training and test sets
        train = data[data["date"] < cutoff_date]
        test = data[data["date"] > cutoff_date]  # Fixed from original code where test was same as train

        # Train model and make predictions
        self.rf.fit(train[predictors], train["target"])
        predictions = self.rf.predict(test[predictors])

        # Combine predictions with actual results
        results = pd.DataFrame({
            "actual": test["target"],
            "prediction": predictions
        })

        # Add match details to results
        results = results.merge(
            data[["date", "team", "opponent", "result"]],
            left_index=True,
            right_index=True
        )

        # Calculate precision score
        precision = precision_score(test["target"], predictions)

        return results, precision

    def standardize_team_names(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize team names for consistency.

        Args:
            results: DataFrame containing match results

        Returns:
            DataFrame with standardized team names
        """
        # Create mapping dictionary that returns original key if not found
        mapping = type('MissingDict', (dict,), {
            '__missing__': lambda self, key: key
        })(self.team_mappings)

        results["new_team"] = results["team"].map(mapping)
        return results


def main():
    """
    Main execution function.
    """
    predictor = FootballPredictor("matches.csv")
    prepared_data = predictor.prepare_features()
    results, precision = predictor.make_predictions(prepared_data)

    # Standardize team names and merge for final analysis
    standardized_results = predictor.standardize_team_names(results)
    final_results = standardized_results.merge(
        standardized_results,
        left_on=["date", "new_team"],
        right_on=["date", "opponent"]
    )

    print(f"Model Precision: {precision:.3f}")
    return final_results


if __name__ == "__main__":
    final_results = main()