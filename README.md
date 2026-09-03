# ⚽ LLM La Liga Predictor

AI-powered La Liga match predictor combining Random Forest ML with LLM reasoning for accurate match outcome predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![ML](https://img.shields.io/badge/ML-RandomForest-orange)
![LLM](https://img.shields.io/badge/LLM-Gemini-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

- Match Prediction - Predicts Home Win / Draw / Away Win with confidence percentages
- Team Statistics - Displays recent form, goals average, and expected goals (xG)
- Head-to-Head Analysis - Shows historical win percentages between any two teams
- Last 5 Meetings - Displays recent match results like FotMob
- Upcoming Matches - Quick-predict panel for upcoming fixtures
- LLM-Powered Insights - Gemini API provides contextual match analysis
- Clean Interface - La Liga themed red and white design
- Real-time Analysis - Instant predictions with visual probability cards

---

## How It Works

The model is trained on 6 seasons of La Liga data (2019-2024) with 2,280 matches. It uses a Random Forest classifier with 9 key features:

- Team Form - Points per match over the last 5 games
- Goals Average - Average goals scored and conceded
- Expected Goals (xG) - Advanced metric for attacking quality
- Head-to-Head - Historical win percentages between teams
- Possession and Shots - Additional statistical indicators

The prediction pipeline processes user-selected teams, calculates all features in real-time, and outputs win probabilities with a final prediction. LLM integration adds contextual insights to explain predictions.

---

## Tech Stack

- Python 3.9+
- Pandas, NumPy - Data Processing
- Scikit-learn - Machine Learning (Random Forest)
- Google Gemini API - LLM Integration
- Streamlit - Web Framework
- Joblib - Model Persistence
- Matplotlib, Plotly - Visualization

---

## Installation

Clone the repository:
git clone https://github.com/YoavBenRaph/llm-la-liga-predictor.git
cd llm-la-liga-predictor

Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run dashboard.py

---

## Model Performance

- Accuracy: ~55-60%
- Features: 9
- Training Data: 2,280 matches
- Seasons: 6 (2019-2024)
- Teams: 25+
- Algorithm: Random Forest

Football prediction is inherently challenging. 55-60% accuracy is competitive for this domain and comparable to expert analysts.

---

## Data Source

The dataset was compiled from historical La Liga match data spanning 6 seasons, including:

- Match results (goals, outcomes)
- Expected goals (xG) and expected goals against (xGA)
- Possession, shots, shots on target
- Team form and head-to-head history
- Over 30 unique data points per match

---

## Future Improvements

- Add live fixtures via API
- Add betting odds comparison
- Implement XGBoost for improved accuracy
- Add more advanced features (player stats, injuries)
- Deploy to Streamlit Cloud

---

## License

This project is licensed under the MIT License.

---

## Author

Your Name
- GitHub: @YoavBenRaph
- LinkedIn: [Yoav Ben Raphael](https://www.linkedin.com/in/yoav-ben-raphael-6464b4367/)

---

## Acknowledgments

- Built with Python, Scikit-learn, and Streamlit
- Inspired by La Liga football
- Data compiled from public sources
