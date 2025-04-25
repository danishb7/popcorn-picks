# **Popcorn Picks: Personalized Movie Recommendation System**

Popcorn Picks is an interactive movie recommendation system designed to deliver personalized movie suggestions based on user preferences and historical ratings. This project combines collaborative filtering with content-based recommendations to create a hybrid recommendation engine. The system is powered by machine learning algorithms and features a user-friendly interface built with Streamlit.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Features](#features)
3. [Data Sources](#data-sources)
4. [Technologies Used](#technologies-used)
5. [Installation and Setup](#installation-and-setup)
6. [How It Works](#how-it-works)
7. [Usage](#usage)
8. [Contributing](#contributing)

---

## **Introduction**
Popcorn Picks uses the MovieLens 25M dataset and additional survey data to create recommendations tailored to the user's preferences. The system incorporates collaborative filtering (using Singular Value Decomposition) and content-based filtering (using user-specific genre preferences) to suggest movies that the user is likely to enjoy.

---

## **Features**
- **Hybrid Recommendation Engine**:
  Combines collaborative filtering and content-based filtering for accurate predictions.
- **Interactive User Interface**:
  Streamlit-based UI allows users to interact with the system using sliders, buttons, and dropdowns.
- **Personalized Recommendations**:
  Recommendations adapt based on user ratings and genre preferences.
- **Efficient Data Handling**:
  Handles large datasets efficiently while ensuring quick response times.
- **Customizable Layouts**:
  Movies are displayed in various formats, including grids, lists, and sliding windows.

---

## **Data Sources**
The following datasets are used in this project:
1. **MovieLens 25M Dataset**:
   - `movies.csv`: Contains movie titles and genres.
   - `ratings.csv`: Contains user ratings for movies.
   - Source: [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)

2. **Survey Data**:
   - A locally collected dataset (`survey.csv`) with anonymized user preferences for specific genres, including Action, Comedy, Sci-Fi, etc.

---

## **Technologies Used**
The project leverages the following technologies and libraries:
- **Python**: Programming language.
- **Pandas**: Data manipulation and preprocessing.
- **NumPy**: Numerical computations.
- **Surprise**: Recommendation system library for collaborative filtering.
- **Streamlit**: Framework for building interactive web applications.
- **Pickle**: For saving and loading the trained model.

---

## **Installation and Setup**
Follow these steps to set up and run the project locally:

### **1. Clone the Repository**
```bash
git clone https://github.com/danishb7/popcorn-picks.git
cd popcorn-picks
```

### **2. Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
Launch the Streamlit app:
```bash
streamlit run app/app3.py
```

---

## **How It Works**
1. **Data Preprocessing**:
   - Genres are split into lists using delimiters.
   - Users with less than 50 ratings are filtered out to ensure robust predictions.
   - The `ratings.csv` and `movies.csv` datasets are merged to include genre information.

2. **Model Training**:
   - The SVD model is trained using the Surprise library on a filtered subset of the ratings data.
   - The model predicts user ratings for movies based on historical data and user preferences.

3. **Recommendation Generation**:
   - Collaborative filtering predictions are combined with genre-based scores for a hybrid recommendation approach.
   - Final scores are calculated as 60% collaborative filtering and 40% content-based scores.

4. **User Interface**:
   - Users interact with the app via sliders and buttons to get personalized recommendations.

---

## **Usage**
1. Launch the app by following the installation steps.
2. Interact with the UI to input your preferences.
3. View personalized movie recommendations in your preferred layout.

---

## **Contributing**
Contributions are welcome! If you have ideas for improvement or want to add new features:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add feature-name"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a Pull Request.

---

## **Acknowledgments**
- [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)
- [Surprise Library](https://surprise.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

Let us know if you have any questions or feedback about Popcorn Picks! Happy movie hunting! 🍿
