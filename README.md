# ⚽ Premier League Match Predictor

A machine learning model that predicts Premier League match outcomes using historical match data and team statistics.

## 🎯 Overview

This project consists of three main components:
1. A web scraper that collects match data from fbref.com
2. A machine learning model that analyzes historical performance
3. A prediction interface for matchup outcomes

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Processing & Analysis**
  - pandas
  - scikit-learn
  - BeautifulSoup4
  - requests
- **Machine Learning**
  - Random Forest Classifier
  - Rolling average features
  - Time-series based validation

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/premier-league-predictor.git

# Navigate to the project directory
cd FootballMatchPredictor

# Install required packages
npm install
```

## 📊 Features

- **Data Collection**
  - Automated web scraping of match statistics
  - Comprehensive team performance metrics
  - Historical match results

- **Match Analysis**
  - Rolling averages of key performance indicators
  - Home/away performance consideration
  - Team form analysis

- **Prediction Model**
  - Win probability calculation
  - Team-specific performance metrics
  - Venue impact consideration

## 🔍 Model Features

The prediction model considers various factors including:
- Goals scored and conceded
- Shots and shots on target
- Average shot distance
- Free kicks and penalties
- Historical head-to-head results
- Home/away advantage

## 📈 Performance

The model's performance is evaluated using precision score, which measures the accuracy of predicted wins. The current model achieves a precision score of approximately 0.63 (varies based on training data).

## 🚀 Future Improvements

- Add player-specific statistics
- Add support for other leagues
- Create a web interface

