import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# PHASE 1: LOAD DATA AND CLEAN DATASET
def load_and_clean_data(file_path):
    print("Data is loading.......")
    df = pd.read_csv(file_path)
    print(df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Fix the warnings by avoiding chained assignments with inplace=True
    # Instead, use direct assignment with loc or update the entire column
    df = df.copy()  # Create a copy to avoid modifying views

    # Fill missing values in numerical columns
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Height'] = df['Height'].fillna(df['Height'].median())
    df['Weight'] = df['Weight'].fillna(df['Weight'].median())

    # Fill missing values in categorical columns
    category_data = ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal',
                     'Fitness Type', 'Exercises', 'Equipment',
                     'Diet (Vegetable)', 'Diet (protein intake)', 'Diet (Juice)']

    for col in category_data:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fixed BMI calculation logic
    if 'BMI' in df.columns:
        mask = df['BMI'].isnull()
        df.loc[mask, 'BMI'] = df.loc[mask, 'Weight'] / ((df.loc[mask, 'Height'] / 100) ** 2)
    else:
        df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)

    if 'BMI Level' not in df.columns:
        df['BMI Level'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

    # Map categorical values to numerical
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})
    df['Hypertension'] = df['Hypertension'].map({'Yes': 1, 'No': 0})
    df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df


# PHASE 2: FEATURE ENGINEERING
def feature_engineering(df):
    df = df.copy()  # Create a copy to avoid modifying views

    # Create age groups
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 18, 35, 50, 65, 100],
        labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
    )

    # Initialize health risk score
    df['Health_Risk_Score'] = 0

    # Update health risk scores - avoiding inplace modifications
    # Instead of incrementing with +=, we'll set the entire column at once
    risk_score = df['Health_Risk_Score'].copy()
    risk_score = np.where(df['Hypertension'] == 1, risk_score + 1, risk_score)
    risk_score = np.where(df['Diabetes'] == 1, risk_score + 1, risk_score)
    risk_score = np.where(df['BMI'] < 18.5, risk_score + 1, risk_score)
    risk_score = np.where(df['BMI'] > 30, risk_score + 1, risk_score)
    risk_score = np.where(df['BMI'] > 35, risk_score + 1, risk_score)
    df['Health_Risk_Score'] = risk_score

    # Set compatibility scores - using loc to avoid warnings
    df['Cardio_Compatibility'] = 1
    df.loc[(df['Hypertension'] == 1) & (df['Age'] > 60), 'Cardio_Compatibility'] = 0.5

    df['Strength_Compatibility'] = 1
    df.loc[(df['BMI'] < 18.5), 'Strength_Compatibility'] = 0.7

    # Set dietary needs
    df['Needs_Low_Sodium'] = (df['Hypertension'] == 1).astype(int)
    df['Needs_Low_Carb'] = (df['Diabetes'] == 1).astype(int)

    if 'Fitness Goal' in df.columns:
        df['Needs_High_Protein'] = (df['Fitness Goal'] == 'Muscle Gain').astype(int)
    else:
        df['Needs_High_Protein'] = 0

    # One-hot encode categorical features
    categorical_columns = []
    if 'BMI Level' in df.columns:
        categorical_columns.append('BMI Level')
    if 'Fitness Type' in df.columns:
        categorical_columns.append('Fitness Type')
    if 'Fitness Goal' in df.columns:
        categorical_columns.append('Fitness Goal')
    categorical_columns.append('Age_Group')

    dummies_df = pd.get_dummies(df[categorical_columns], prefix=categorical_columns)
    df_numeric = df.drop(categorical_columns, axis=1)
    df_final = pd.concat([df_numeric, dummies_df], axis=1)

    print(f"Dataset shape after feature engineering: {df_final.shape}")
    print(f"New features added: {df_final.columns.tolist()}")
    return df_final


# PHASE 3: MODEL DEVELOPMENT
def prepare_training_data(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'ID' in numerical_features:
        numerical_features.remove('ID')
    if 'Recommendation' in numerical_features:
        numerical_features.remove('Recommendation')

    dummy_feature = [col for col in df.columns if
                     any(col.startswith(prefix) for prefix in
                         ['BMI Level_', 'Fitness Goal_', 'Fitness Type_', 'Age_Group_'])]
    feature = numerical_features + dummy_feature

    if 'Recommendation' in df.columns:
        x = df[feature]
        y = df['Recommendation']

        # split data in train and test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test
    else:
        print("Error: Recommendation column not found.")
        return None, None, None, None


def train_model(x_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    x_train = x_train.astype('float32')
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    x_test = x_test.astype('float32')
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Add zero_division parameter to avoid warnings
    print(classification_report(y_test, y_pred, zero_division=0))

    # Feature importance
    feature_importances = pd.DataFrame({
        'Feature': x_test.columns,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importances.head(10))

    return accuracy, feature_importances


# PHASE 4: RECOMMENDATION PIPELINE
def preprocess_input(user_data, feature_names):
    if not isinstance(user_data, pd.DataFrame):
        user_data = pd.DataFrame([user_data])  # Convert dict to DataFrame if needed

    # Calculate BMI if not present
    if 'BMI' not in user_data.columns and 'Height' in user_data.columns and 'Weight' in user_data.columns:
        user_data['BMI'] = user_data['Weight'] / ((user_data['Height'] / 100) ** 2)

    # Initialize and calculate health risk score
    user_data['Health_Risk_Score'] = 0

    # Use cleaner approach for risk score calculation
    risk_score = user_data['Health_Risk_Score'].copy()

    if 'Hypertension' in user_data.columns:
        risk_score = np.where(user_data['Hypertension'] == 1, risk_score + 1, risk_score)
    if 'Diabetes' in user_data.columns:
        risk_score = np.where(user_data['Diabetes'] == 1, risk_score + 1, risk_score)
    if 'BMI' in user_data.columns:
        risk_score = np.where(user_data['BMI'] < 18.5, risk_score + 1, risk_score)
        risk_score = np.where(user_data['BMI'] > 30, risk_score + 1, risk_score)
        risk_score = np.where(user_data['BMI'] > 35, risk_score + 1, risk_score)

    user_data['Health_Risk_Score'] = risk_score

    # Set compatibility scores
    user_data['Cardio_Compatibility'] = 1
    if 'Hypertension' in user_data.columns and 'Age' in user_data.columns:
        user_data.loc[(user_data['Hypertension'] == 1) & (user_data['Age'] > 60), 'Cardio_Compatibility'] = 0.5

    user_data['Strength_Compatibility'] = 1
    if 'BMI' in user_data.columns:
        user_data.loc[(user_data['BMI'] < 18.5), 'Strength_Compatibility'] = 0.7

    # Set dietary needs
    if 'Hypertension' in user_data.columns:
        user_data['Needs_Low_Sodium'] = (user_data['Hypertension'] == 1).astype(int)
    if 'Diabetes' in user_data.columns:
        user_data['Needs_Low_Carb'] = (user_data['Diabetes'] == 1).astype(int)
    if 'Fitness Goal' in user_data.columns:
        user_data['Needs_High_Protein'] = (user_data['Fitness Goal'] == 'Muscle Gain').astype(int)

    # Create age group if age is available
    if 'Age' in user_data.columns:
        user_data['Age_Group'] = pd.cut(
            user_data['Age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
        )

    # One-hot encode categorical features
    categorical_columns = []
    if 'Age_Group' in user_data.columns:
        categorical_columns.append('Age_Group')
    if 'BMI Level' in user_data.columns:
        categorical_columns.append('BMI Level')
    if 'Fitness Goal' in user_data.columns:
        categorical_columns.append('Fitness Goal')
    if 'Fitness Type' in user_data.columns:
        categorical_columns.append('Fitness Type')

    # Create one-hot encoding for each category
    for col in categorical_columns:
        if col in user_data.columns:
            dummies = pd.get_dummies(user_data[col], prefix=col)
            user_data = pd.concat([user_data, dummies], axis=1)
            user_data = user_data.drop(col, axis=1)

    # Create a DataFrame with all required feature columns
    processed_data = pd.DataFrame(columns=feature_names)

    # Fill in available features
    for feature in feature_names:
        if feature in user_data.columns:
            processed_data[feature] = user_data[feature]
        else:
            processed_data[feature] = 0  # Default value for missing features

    # Convert to float to ensure compatibility with the model
    processed_data = processed_data.astype(float)

    return processed_data


def generate_recommendations(model, user_data, feature_names):
    processed_data = preprocess_input(user_data, feature_names)
    recommendation = model.predict(processed_data)[0]
    probabilities = model.predict_proba(processed_data)
    confidence = np.max(probabilities)

    return recommendation, confidence


def explain_recommendation(recommendation, user_data):
    explanation = f"Based on your profile, we recommend: {recommendation}\n\n"
    health_conditions = []
    if user_data.get('Hypertension', 0) == 1:
        health_conditions.append("hypertension")
    if user_data.get('Diabetes', 0) == 1:
        health_conditions.append("diabetes")

    if health_conditions:
        explanation += f"This recommendation takes into account your health conditions: {', '.join(health_conditions)}.\n"

    bmi = user_data.get('BMI', None)
    if bmi is not None:
        if bmi < 18.5:
            explanation += "As your BMI indicates you're underweight, we've focused on nutrition that supports healthy weight gain.\n"
        elif bmi >= 25 and bmi < 30:
            explanation += "As your BMI indicates you're overweight, we've included elements to support healthy weight management.\n"
        elif bmi >= 30:
            explanation += "As your BMI indicates obesity, we've prioritized gentle exercise options and nutritional guidance.\n"

    age = user_data.get('Age', None)
    if age is not None:
        if age < 18:
            explanation += "This plan is tailored for teenagers, focusing on development and establishing healthy habits.\n"
        elif age > 65:
            explanation += "This plan is adapted for seniors, emphasizing low-impact activities and joint health.\n"

    fitness_goal = user_data.get('Fitness Goal', None)
    if fitness_goal:
        explanation += f"Your goal of '{fitness_goal}' has shaped the core of this recommendation.\n"

    return explanation


def safety_check(recommendation, user_data):
    warnings = []

    # Check for high-intensity recommendations for people with heart conditions
    if (user_data.get('Hypertension', 0) == 1 and
            user_data.get('Age', 0) > 60 and
            "high intensity" in recommendation.lower()):
        warnings.append(
            "CAUTION: High-intensity exercises should be approached with care given your hypertension. Consult your doctor.")

    # Check for high-impact exercises for obese individuals
    if (user_data.get('BMI', 0) > 35 and
            any(x in recommendation.lower() for x in ["jumping", "running", "high impact"])):
        warnings.append(
            "CAUTION: High-impact exercises may strain your joints. Consider low-impact alternatives like swimming.")

    # Check for diabetes-specific nutritional guidance
    if (user_data.get('Diabetes', 0) == 1 and
            not any(x in recommendation.lower() for x in ["low carb", "low sugar", "glycemic index"])):
        warnings.append(
            "NOTE: Since you have diabetes, pay special attention to carbohydrate intake and monitor blood sugar levels.")

    return warnings


def get_user_input():
    user_data = {}

    # Get gender input
    while True:
        gender = input("Enter your gender (Male/Female): ").strip().capitalize()
        if gender in ['Male', 'Female']:
            user_data['Sex'] = 1 if gender == 'Male' else 0
            break
        else:
            print("Invalid input. Please enter 'Male' or 'Female'.")

    # Get age input
    while True:
        try:
            age = int(input("Enter your age (years): "))
            if 10 <= age <= 100:
                user_data['Age'] = age
                break
            else:
                print("Please enter a valid age between 10 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get height input
    while True:
        try:
            height = float(input("Enter your height (cm): "))
            if 100 <= height <= 220:
                user_data['Height'] = height
                break
            else:
                print("Please enter a valid height between 100 and 220 cm.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get weight input
    while True:
        try:
            weight = float(input("Enter your weight (kg): "))
            if 30 <= weight <= 250:
                user_data['Weight'] = weight
                break
            else:
                print("Please enter a valid weight between 30 and 250 kg.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Calculate BMI automatically
    bmi = weight / ((height / 100) ** 2)
    user_data['BMI'] = bmi
    print(f"Your calculated BMI is: {bmi:.2f}")

    # Health conditions
    while True:
        hypertension = input("Do you have hypertension (Yes/No)? ").strip().capitalize()
        if hypertension in ['Yes', 'No']:
            user_data['Hypertension'] = 1 if hypertension == 'Yes' else 0
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")

    while True:
        diabetes = input("Do you have diabetes (Yes/No)? ").strip().capitalize()
        if diabetes in ['Yes', 'No']:
            user_data['Diabetes'] = 1 if diabetes == 'Yes' else 0
            break
        else:
            print("Invalid input. Please enter 'Yes' or 'No'.")

    # Fitness goals
    print("\nFitness Goals:")
    print("1. Weight Loss")
    print("2. Muscle Gain")
    print("3. Cardiovascular Health")
    print("4. Flexibility")
    print("5. General Fitness")

    goals = {
        '1': 'Weight Loss',
        '2': 'Muscle Gain',
        '3': 'Cardiovascular Health',
        '4': 'Flexibility',
        '5': 'General Fitness'
    }

    while True:
        goal_choice = input("Choose your fitness goal (1-5): ").strip()
        if goal_choice in goals:
            user_data['Fitness Goal'] = goals[goal_choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    # Fitness types
    print("\nPreferred Exercise Type:")
    print("1. Cardio")
    print("2. Strength Training")
    print("3. Yoga/Pilates")
    print("4. HIIT")
    print("5. Mixed")

    types = {
        '1': 'Cardio Fitness',
        '2': 'Muscular Fitness',
        '3': 'Yoga/Pilates',
        '4': 'HIIT',
        '5': 'Mixed'
    }

    while True:
        type_choice = input("Choose your preferred exercise type (1-5): ").strip()
        if type_choice in types:
            user_data['Fitness Type'] = types[type_choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    # Add BMI Level based on calculated BMI
    if bmi < 18.5:
        user_data['BMI Level'] = 'Underweight'
    elif bmi < 25:
        user_data['BMI Level'] = 'Normal'
    elif bmi < 30:
        user_data['BMI Level'] = 'Overweight'
    else:
        user_data['BMI Level'] = 'Obese'

    return user_data


def build_recommendation_system(data_path):
    # Phase 1: Data loading and cleaning
    df = load_and_clean_data(data_path)

    # Phase 2: Feature engineering
    df = feature_engineering(df)

    # Phase 3: Model development
    X_train, X_test, y_train, y_test = prepare_training_data(df)
    if X_train is None:
        return None, None

    model = train_model(X_train, y_train)
    accuracy, feature_importance = evaluate_model(model, X_test, y_test)

    # Return model and feature names for use in the recommendation pipeline
    return model, X_train.columns.tolist()


def get_recommendation_for_user(model, feature_names, user_data):
    recommendation, confidence = generate_recommendations(model, user_data, feature_names)
    explanation = explain_recommendation(recommendation, user_data)
    warnings = safety_check(recommendation, user_data)

    result = {
        'recommendation': recommendation,
        'confidence': confidence,
        'explanation': explanation,
        'warnings': warnings
    }

    return result


# Main execution block
if __name__ == "__main__":
    # Example path - replace with your actual file path
    data_path = "./Dataset/gym recommendation (1).csv"

    # Build the recommendation system
    model, feature_names = build_recommendation_system(data_path)

    if model is not None:
        # Get user input
        user_data = get_user_input()

        # Get recommendation
        result = get_recommendation_for_user(model, feature_names, user_data)

        # Display results
        print("\nRecommendation Results:")
        print("=====================")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"\nExplanation:\n{result['explanation']}")

        if result['warnings']:
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"- {warning}")
    else:
        print("Failed to build recommendation system. Please check your data.")