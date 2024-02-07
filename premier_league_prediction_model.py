import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Function to fetch data from The Odds API
def fetch_data():
    api_key = '85174e5181a42ac5640cca267ae2ab80'  # Replace 'YOUR_API_KEY' with your actual API key
    url = f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={api_key}&regions=uk&eu&markets=h2h'
    response = requests.get(url)
    data = response.json()
    return data

# Function to preprocess the fetched data
def preprocess_data(data):
    matches = []
    for match in data:
        home_team = match['home_team']
        away_team = match['away_team']
        home_odds = None
        away_odds = None
        draw_odds = None
        for bookmaker in match['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == home_team:
                            home_odds = outcome['price']
                        elif outcome['name'] == away_team:
                            away_odds = outcome['price']
                        elif outcome['name'] == 'Draw':
                            draw_odds = outcome['price']
        if home_odds is not None and away_odds is not None and draw_odds is not None:
            # Assign label: 1 if home team wins, 0 if draw, -1 if away team wins
            label = 1 if home_odds < away_odds and home_odds < draw_odds else 0 if draw_odds < home_odds and draw_odds < away_odds else -1
            matches.append({'Home Team': home_team, 'Away Team': away_team, 'Home Odds': home_odds, 'Away Odds': away_odds, 'Draw Odds': draw_odds, 'Label': label})
    return pd.DataFrame(matches)

# Function to train the logistic regression model
def train_model(df):
    X = df[['Home Odds', 'Away Odds', 'Draw Odds']]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Main function
def main():
    # Fetch data from The Odds API
    data = fetch_data()
    # Preprocess the data
    df = preprocess_data(data)
    # Train the model
    model, X_test, y_test = train_model(df)
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    # Save the trained model
    joblib.dump(model, 'trained_model.h5')

if __name__ == "__main__":
    main()