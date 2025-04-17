import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample data
data = {
    'age': [25, 30, 22, 28],
    'gender': ['M', 'F', 'F', 'M'],
    'steps_per_day': [5000, 8000, 3000, 10000],
    'heart_rate': [80, 75, 90, 70],
    'goal': ['weight_loss', 'maintain', 'weight_gain', 'weight_loss']
}
df = pd.DataFrame(data)

# Encode text to numbers
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_goal = LabelEncoder()
df['goal'] = le_goal.fit_transform(df['goal'])

# Split input and target
X = df.drop('goal', axis=1)
y = df['goal']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🏃 Personal Fitness Tracker AI")
st.write("Enter your fitness info to get a personalized goal & workout suggestion!")

age = st.slider("Your Age", 10, 70, 25)
gender = st.selectbox("Your Gender", ['M', 'F'])
steps = st.slider("Average Steps Per Day", 1000, 20000, 7000)
heart_rate = st.slider("Resting Heart Rate", 50, 120, 75)

# Prepare user input
user_input = pd.DataFrame([[age, le_gender.transform([gender])[0], steps, heart_rate]],
                          columns=['age', 'gender', 'steps_per_day', 'heart_rate'])

# Predict
goal_pred = model.predict(user_input)[0]

# Decode goal
goal_label = le_goal.inverse_transform([goal_pred])[0]

# Suggest workout
def suggest_workout(goal):
    return {
        'weight_loss': "🔥 Do HIIT or Cardio",
        'maintain': "💪 Mix of Strength + Light Cardio",
        'weight_gain': "🏋️ Focus on Weight Training and Calories"
    }.get(goal_label, "🏃 Stay active and healthy!")

# Show results
st.subheader("🎯 Predicted Fitness Goal:")
st.success(goal_label)

st.subheader("💡 Suggested Workout Plan:")
st.info(suggest_workout(goal_label))
