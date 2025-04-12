import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from unicodedata import category


#PHASE 1  LOAD DATA AND CLEAN DATASET

def load_and_clean_data(file_path):
    print("Data is loading.......")
    df =pd.read_csv(file_path)
    print(df.shape)
    print("\n for missing value before cleaning")
    print(df.isnull().sum())
    df['Age'].fillna(df['Age'].median(),inplace=True)
    df['Height'].fillna(df['Height'].median(), inplace=True)
    df['Weight'].fillna(df['Weight'].median(), inplace=True)
    category_data = ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal',
                        'Fitness Type', 'Exercises', 'Equipment',
                        'Diet (Vegetable)', 'Diet (protein intake)', 'Diet (Juice)']

    for col in category_data:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0],inplace= True)

    if 'BMI' not in df.columns:
        mask = df['BMI'].isnull()
        df.loc[mask,'BMI'] =  df.loc[mask, 'Weight'] / ((df.loc[mask, 'Height'] / 100) ** 2)
    else:
        df['BMI'] = df['Weight']/((df['Height']/100)**2)
    if 'BMI Level' not in df.columns:
        df['BMI Level'] = pd.cut(
            df['BMI'],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
    df['Sex']=df['Sex'].map({'Male': 1, 'Female': 0})
    df['Hypertension'] = df['Hypertension'].map({'Yes': 1, 'No': 0})
    df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    return df

#PHASR 2: FEATURE ENGINEERING

def  feature_engineering(df):
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0,18,35,50,65,100],
        labels=['Teen','Young_Adult','Adult','Middle_Age','Senior']
    )
    df['Health_Risk_Score' ] = 0
    df.loc[df['Hypertension'] == 1, 'Health_Risk_Score'] += 1
    df.loc[df['Diabetes'] == 1, 'Health_Risk_Score'] += 1
    df.loc[df['BMI'] < 18.5, 'Health_Risk_Score'] += 1
    df.loc[df['BMI'] > 30, 'Health_Risk_Score'] += 1
    df.loc[df['BMI'] > 35, 'Health_Risk_Score'] += 1

    df['Cardio_Compatibility'] = 1
    df.loc[(df['Hypertension'] == 1) & (df['Age'] > 60), 'Cardio_Compatibility'] = 0.5
    df['Strength_Compatibility'] = 1
    df.loc[(df['BMI'] < 18.5), 'Strength_Compatibility'] = 0.7
    df['Needs_Low_Sodium'] = df['Hypertension'] == 1
    df['Needs_Low_Carb'] = df['Diabetes'] == 1

    if 'Fitness Goal' in df.columns:
        df['Needs_High_Protein'] = (df['Fitness Goal'] == 'Muscle Gain').astype(int)
    else:
        df['Needs_High_Protein'] = 0

    categorical_columns=[]
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


#PHASE 3: MODEL DEVELOPMENT

def prepare_training_data(df):
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'ID' in numerical_features:
        numerical_features.remove('ID')
    if 'Recommendation' in numerical_features:
        numerical_features.remove('Recommendation')
    dummy_feature = [col for col in df.columns if col.startswith('BMI_','Fitness Goal_', 'Fitness Type_', 'Age_Group_')]
    feature = numerical_features + dummy_feature
    # print(feature)

    if 'Recommendation' in df.columns:
        x = df[feature]
        y = df['Recommendation']

        #split data in train and test
        x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test
    else:
        print("Error: column not found.")
        return None, None, None, None
def train_model(x_train,y_train):
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
    print(accuracy)
    print("report")
    print(classification_report(y_test, y_pred))

    #feature importance
    feature_importances = pd.DataFrame({
        'Feature': x_test.columns,
        'Importance': model.feature_importances_,

    }).sort_values('Importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importances.head(10))

    return accuracy, feature_importances

def preprocess_input(user_data, feature_names):
    if not isinstance(user_data, pd.DataFrame):
        user_data = pd.DataFrame(user_data)
    if 'BMI' not in user_data.columns  and 'Height' in user_data.columns and 'Weight' in user_data.columns:
        user_data['BMI'] = user_data['Weight'] / ((user_data['Height'] / 100) ** 2)
    user_data['Health_Risk_Score'] = 0
    if 'Hypertension' in user_data.columns:
        user_data.loc[user_data['Hypertension'] == 1, 'Health_Risk_Score'] += 1
    if 'Diabetes' in user_data.columns:
        user_data.loc[user_data['Diabetes'] == 1, 'Health_Risk_Score'] += 1
    if 'BMI' in user_data.columns:
        user_data.loc[user_data['BMI'] < 18.5, 'Health_Risk_Score'] += 1
        user_data.loc[user_data['BMI'] > 30, 'Health_Risk_Score'] += 1
        user_data.loc[user_data['BMI'] > 35, 'Health_Risk_Score'] += 1
    user_data['Cardio_Compatibility'] = 1
    if 'Hypertension' in user_data.columns and 'Age' in user_data.columns:
        user_data.loc[(user_data['Hypertension'] == 1) & (user_data['Age'] > 60), 'Cardio_Compatibility'] = 0.5

    user_data['Strength_Compatibility'] = 1
    if 'BMI' in user_data.columns:
        user_data.loc[(user_data['BMI'] < 18.5), 'Strength_Compatibility'] = 0.7
    user_data['Strength_Compatibility'] = 1
    if 'BMI' in user_data.columns:
        user_data.loc[(user_data['BMI'] < 18.5), 'Strength_Compatibility'] = 0.7

    if 'Hypertension' in user_data.columns:
        user_data['Needs_Low_Sodium'] = user_data['Hypertension'] == 1
    if 'Diabetes' in user_data.columns:
        user_data['Needs_Low_Carb'] = user_data['Diabetes'] == 1
    if 'Fitness Goal' in user_data.columns:
        user_data['Needs_High_Protein'] = (user_data['Fitness Goal'] == 'Muscle Gain').astype(int)
    if 'Age' in user_data.columns:
        user_data['Age_Group'] = pd.cut(
            user_data['Age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
        )

        # One-hot encode Age_Group and other categorical features
        categorical_columns = ['Age_Group']

        if 'BMI Level' in user_data.columns:
            categorical_columns.append('BMI Level')

        if 'Fitness Goal' in user_data.columns:
            categorical_columns.append('Fitness Goal')

        if 'Fitness Type' in user_data.columns:
            categorical_columns.append('Fitness Type')

        for col in categorical_columns:
            if col in user_data.columns:
                # Create one-hot encoding for each category
                dummies = pd.get_dummies(user_data[col], prefix=col)
                user_data = pd.concat([user_data, dummies], axis=1)
                user_data = user_data.drop(col, axis=1)
                processed_data = pd.DataFrame(columns=feature_names)

                for feature in feature_names:
                    if feature in user_data.columns:
                        processed_data[feature] = user_data[feature]
                    else:
                        processed_data[feature] = 0  # Default value for missing features

                # Convert to float to ensure compatibility with the model
                processed_data = processed_data.astype(float)

                return processed_data



#
# def get_user_input():
#
#     user_data = {}
#     while True:
#         gender = input("Enter your gender (Male/Female): ").strip().capitalize()
#         if gender in ['Male', 'Female']:
#             user_data['Sex'] = 1 if gender == 'Male' else 0
#             break
#         else:
#             print("Invalid input. Please enter 'Male' or 'Female'.")
#
#     # Get age input
#     while True:
#         try:
#             age = int(input("Enter your age (years): "))
#             if 10 <= age <= 100:
#                 user_data['Age'] = age
#                 break
#             else:
#                 print("Please enter a valid age between 10 and 100.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")
#
#     # Get height input
#     while True:
#         try:
#             height = float(input("Enter your height (cm): "))
#             if 100 <= height <= 220:
#                 user_data['Height'] = height
#                 break
#             else:
#                 print("Please enter a valid height between 100 and 220 cm.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")
#
#     # Get weight input
#     while True:
#         try:
#             weight = float(input("Enter your weight (kg): "))
#             if 30 <= weight <= 250:
#                 user_data['Weight'] = weight
#                 break
#             else:
#                 print("Please enter a valid weight between 30 and 250 kg.")
#         except ValueError:
#             print("Invalid input. Please enter a number.")
#
#     # Calculate BMI automatically
#     bmi = weight / ((height / 100) ** 2)
#     user_data['BMI'] = bmi
#     print(f"Your calculated BMI is: {bmi:.2f}")
#
#     # Health conditions
#     while True:
#         hypertension = input("Do you have hypertension (Yes/No)? ").strip().capitalize()
#         if hypertension in ['Yes', 'No']:
#             user_data['Hypertension'] = 1 if hypertension == 'Yes' else 0
#             break
#         else:
#             print("Invalid input. Please enter 'Yes' or 'No'.")
#
#     while True:
#         diabetes = input("Do you have diabetes (Yes/No)? ").strip().capitalize()
#         if diabetes in ['Yes', 'No']:
#             user_data['Diabetes'] = 1 if diabetes == 'Yes' else 0
#             break
#         else:
#             print("Invalid input. Please enter 'Yes' or 'No'.")
#
#     # Fitness goals
#     print("\nFitness Goals:")
#     print("1. Weight Loss")
#     print("2. Muscle Gain")
#     print("3. Cardiovascular Health")
#     print("4. Flexibility")
#     print("5. General Fitness")
#
#     goals = {
#         '1': 'Weight Loss',
#         '2': 'Muscle Gain',
#         '3': 'Cardiovascular Health',
#         '4': 'Flexibility',
#         '5': 'General Fitness'
#     }
#
#     while True:
#         goal_choice = input("Choose your fitness goal (1-5): ").strip()
#         if goal_choice in goals:
#             user_data['Fitness Goal'] = goals[goal_choice]
#             break
#         else:
#             print("Invalid choice. Please enter a number between 1 and 5.")
#
#     # Fitness types
#     print("\nPreferred Exercise Type:")
#     print("1. Cardio")
#     print("2. Strength Training")
#     print("3. Yoga/Pilates")
#     print("4. HIIT")
#     print("5. Mixed")
#
#     types = {
#         '1': 'Cardio',
#         '2': 'Strength Training',
#         '3': 'Yoga/Pilates',
#         '4': 'HIIT',
#         '5': 'Mixed'
#     }
#
#     while True:
#         type_choice = input("Choose your preferred exercise type (1-5): ").strip()
#         if type_choice in types:
#             user_data['Fitness Type'] = types[type_choice]
#             break
#         else:
#             print("Invalid choice. Please enter a number between 1 and 5.")
#
#     # Add BMI Level based on calculated BMI
#     if bmi < 18.5:
#         user_data['BMI Level'] = 'Underweight'
#     elif bmi < 25:
#         user_data['BMI Level'] = 'Normal'
#     elif bmi < 30:
#         user_data['BMI Level'] = 'Overweight'
#     else:
#         user_data['BMI Level'] = 'Obese'
#
#     print("\nThank you for providing your information. Generating recommendation...")
#     return user_data


#Final Main Method
if __name__ == "__main__":
    # Example path - replace with your actual file path
    data_path = "./Dataset/gym recommendation (1).csv"