import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer

df = pd.read_csv('insurance.csv')
# print(df.head())

# we make one copy of original dataset because we do multiple operation on it.
df_feat = df.copy()

# Feature 1 Calculate bmi
df_feat["bmi"] = df_feat["weight"] / (df_feat["height"] **2)

# Feature 2 Age Group
def age_group(age):
    if age < 25:
        return "young"
    elif age < 45 :
        return "middle-aged"
    elif age < 60:
        return "adult"
    else:
        return "senior"

df_feat["age_group"] = df_feat["age"].apply(age_group)

def Lifestyle(row):
    if row["smoker"]  and row["bmi"] > 30:
        return "high"
    elif row["smoker"]  and  row["bmi"]  > 20:
        return "middle"
    else:
        return "low"
df_feat["life_style_risk"] = df_feat.apply(Lifestyle, axis= 1)

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
    ]

# Feature 4: City Tier
def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3

df_feat["city_tier"] = df_feat["city"].apply(city_tier)
df_feat.drop(columns=['age', 'weight', 'height', 'smoker', 'city'])[['income_lpa', 'occupation', 'bmi', 'age_group', 'life_style_risk', 'city_tier', 'insurance_premium_category']].sample(5)
# print(df_feat.head())

X = df_feat.drop(columns=['insurance_premium_category'])
y = df_feat['insurance_premium_category']

# Define Categorical Feature and Numerical Feature
categorical_feature = ["occupation", "age_group", "life_style_risk", "city_tier"]
numerical_feature = ["income_lpa", "bmi"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(), categorical_feature),
    ("num", "passthrough", numerical_feature),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

pickle_model_path = "model.pkl"
with open(pickle_model_path, "wb") as model:
    pickle.dump(pipeline, model)

