import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv('data/crime_data.csv')  # Ensure this file is in the same folder

# Step 2: Define categorical columns to encode
categorical_columns = ['location_type', 'time_of_day', 'day_of_week',
                       'weapon_involved', 'known_offender', 'prior_incidents']

# # ADD THIS LINE BELOW:
# df['num_suspects'] = df['num_suspects'].astype(int)  # or float if needed

# # Then add 'num_suspects' to features:
# X = df[categorical_columns + ['num_suspects']]  # ✅ Now 7 features

# Step 3: Encode categorical features
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Step 4: Define feature columns (INCLUDE num_suspects)
feature_columns = categorical_columns + ['num_suspects']
X = df[feature_columns]

# Step 5: Encode the target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['crime_type'])
label_encoders['crime_type'] = target_encoder

# Step 6: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 9: Save model and encoders
# Save model using joblib
joblib.dump(model, 'crime_prediction_model.joblib')  # ✅ Required name

# Save encoders using joblib instead of pickle
joblib.dump(label_encoders, 'label_encoders.joblib')  # ✅ Required name


print("✅ Training complete. Model and encoders saved.")
