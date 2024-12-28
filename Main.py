from MatchPredictor import FootballPredictor
import pandas as pd
from datetime import datetime


def predict_match(team1: str, team2: str, data_path: str = "matches.csv") -> dict:
    """
    Predict the outcome of a match between two teams.

    Args:
        team1: Name of the home team
        team2: Name of the away team
        data_path: Path to the historical match data CSV file

    Returns:
        Dictionary containing prediction details and win probabilities
    """
    # Initialize predictor
    predictor = FootballPredictor(data_path)

    # Prepare all features
    prepared_data = predictor.prepare_features()

    # Get the most recent data for both teams
    team1_recent = prepared_data[prepared_data['team'] == team1].iloc[-1]
    team2_recent = prepared_data[prepared_data['team'] == team2].iloc[-1]

    # Create a new match entry for prediction
    new_match = pd.DataFrame({
        'date': datetime.now(),
        'team': team1,
        'opponent': team2,
        'venue': 'Home',  # Assuming team1 is home team
        'venue_code': 1,  # Code for home venue
        'opp_code': prepared_data[prepared_data['team'] == team2]['opp_code'].iloc[0],
        'hour': 15,  # Default to 3 PM kickoff
        'day_code': datetime.now().weekday(),
    }, index=[0])

    # Copy rolling averages from most recent matches
    for col in predictor.stat_columns:
        new_match[f'{col}_rolling'] = team1_recent[f'{col}_rolling']

    # Get feature columns in correct order
    predictors = [
                     "venue_code",
                     "opp_code",
                     "hour",
                     "day_code"
                 ] + [f"{col}_rolling" for col in predictor.stat_columns]

    # Train model on all available data
    predictor.rf.fit(prepared_data[predictors], prepared_data['target'])

    # Make prediction
    win_probability = predictor.rf.predict_proba(new_match[predictors])[0][1]

    # Return prediction details
    return {
        'home_team': team1,
        'away_team': team2,
        'home_win_probability': round(win_probability * 100, 2),
        'away_win_probability': round((1 - win_probability) * 100, 2),
        'predicted_winner': team1 if win_probability > 0.5 else team2
    }


def main():
    """
    Command line interface for match prediction.
    """
    print("Welcome to the Football Match Predictor!")
    print("Enter the names of two teams to predict the match outcome.")
    print("Note: Team names must match exactly as they appear in the dataset.")

    while True:
        try:
            team1 = input("\nEnter home team name: ")
            team2 = input("Enter away team name: ")

            result = predict_match(team1, team2)

            print("\nPrediction Results:")
            print(f"Home Team: {result['home_team']}")
            print(f"Away Team: {result['away_team']}")
            print(f"Home Win Probability: {result['home_win_probability']}%")
            print(f"Away Win Probability: {result['away_win_probability']}%")
            print(f"Predicted Winner: {result['predicted_winner']}")

            another = input("\nPredict another match? (y/n): ")
            if another.lower() != 'y':
                break

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please check team names and try again.")


if __name__ == "__main__":
    main()