from flask import Flask, render_template, request, session, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os


app = Flask(__name__)
app.secret_key = 'fitness_recommendation_secret_key'

# Global variables to store model components
original_df = None
knn_model = None
features = None
scaler = None

# Initialize model at startup
model_initialized = False


# PHASE 1: LOAD DATA, EXPLORE AND CLEAN DATASET
def load_and_explore_data(file_path):
    """Load and clean the dataset from a CSV file."""
    app.logger.info("Data is loading...")

    try:
        # Try to infer separator automatically
        df = pd.read_csv(file_path)

        # If the data didn't load properly (only one column), try tab separator
        if len(df.columns) == 1:
            df = pd.read_csv(file_path, sep='\t')

        app.logger.info(f"Dataset loaded successfully with shape: {df.shape}")

        # Convert height from meters to cm if necessary
        if 'Height' in df.columns and df['Height'].max() < 3:  # If height is in meters
            df['Height'] = df['Height'] * 100
            app.logger.info("Height converted from meters to centimeters")

        # Check for duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            app.logger.info(f"Found {dup_count} duplicate rows - removing duplicates")
            df = df.drop_duplicates()

        # Create a copy to avoid modifying views
        df = df.copy()

        # Check missing values
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            app.logger.info(f"Found {missing_before} missing values - cleaning data")

            # Fill missing values in numerical columns
            for col in df.select_dtypes(include=['number']).columns:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())

            # Fill missing values in categorical columns
            category_cols = df.select_dtypes(include=['object']).columns
            for col in category_cols:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        # Handle BMI calculation
        if 'BMI' not in df.columns and 'Height' in df.columns and 'Weight' in df.columns:
            df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
            app.logger.info("BMI calculated from height and weight")
        elif 'BMI' in df.columns:
            # Fill missing BMI values
            mask = df['BMI'].isnull()
            if mask.any() and 'Height' in df.columns and 'Weight' in df.columns:
                df.loc[mask, 'BMI'] = df.loc[mask, 'Weight'] / ((df.loc[mask, 'Height'] / 100) ** 2)

        # Use 'Level' column as BMI level if available, otherwise calculate
        if 'Level' in df.columns:
            df['BMI Level'] = df['Level']
        elif 'BMI' in df.columns:
            # Add BMI Level if not present
            df['BMI Level'] = pd.cut(
                df['BMI'],
                bins=[0, 18.5, 24.9, 29.9, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
            app.logger.info("BMI Level categorized based on BMI values")

        # Map categorical values to numerical for model training if present
        if 'Sex' in df.columns:
            sex_mapping = {'Male': 1, 'Female': 0}
            df['Sex'] = df['Sex'].map(lambda x: sex_mapping.get(x, np.nan))
            df['Sex'] = df['Sex'].fillna(df['Sex'].median())

        if 'Hypertension' in df.columns:
            yn_mapping = {'Yes': 1, 'No': 0}
            df['Hypertension'] = df['Hypertension'].map(lambda x: yn_mapping.get(x, np.nan))
            df['Hypertension'] = df['Hypertension'].fillna(0)  # Default to no hypertension

        if 'Diabetes' in df.columns:
            df['Diabetes'] = df['Diabetes'].map(lambda x: yn_mapping.get(x, np.nan))
            df['Diabetes'] = df['Diabetes'].fillna(0)  # Default to no diabetes

        # Check final missing values
        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            app.logger.warning(f"Warning: {missing_after} missing values remain after cleaning")
        else:
            app.logger.info("Data cleaning complete - no missing values remain")

        return df

    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        return None


# PHASE 2: FEATURE ENGINEERING
def feature_engineering(df):
    """Add new features to improve recommendation quality."""
    if df is None:
        app.logger.error("No data available for feature engineering")
        return None

    app.logger.info("Performing feature engineering...")
    df = df.copy()

    # Create age groups if Age column exists
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(
            df['Age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
        )
        app.logger.info("Age groups created")

    # Initialize health risk score
    df['Health_Risk_Score'] = 0
    risk_score = df['Health_Risk_Score'].copy()

    # Update health risk scores based on available fields
    if 'Hypertension' in df.columns:
        risk_score = np.where(df['Hypertension'] == 1, risk_score + 1, risk_score)

    if 'Diabetes' in df.columns:
        risk_score = np.where(df['Diabetes'] == 1, risk_score + 1, risk_score)

    if 'BMI' in df.columns:
        risk_score = np.where(df['BMI'] < 18.5, risk_score + 1, risk_score)
        risk_score = np.where(df['BMI'] > 30, risk_score + 1, risk_score)
        risk_score = np.where(df['BMI'] > 35, risk_score + 1, risk_score)

    df['Health_Risk_Score'] = risk_score
    app.logger.info("Health risk scores calculated")

    # One-hot encode categorical features
    categorical_columns = []

    # Check if columns exist before adding them to encoding list
    for col in ['BMI Level', 'Age_Group', 'Fitness Type', 'Fitness Goal']:
        if col in df.columns:
            categorical_columns.append(col)

    # Only apply one-hot encoding if we have categorical columns
    if categorical_columns:
        app.logger.info(f"One-hot encoding categorical features: {', '.join(categorical_columns)}")
        dummies_df = pd.get_dummies(df[categorical_columns], prefix=categorical_columns)
        df_numeric = df.drop(categorical_columns, axis=1)
        df_final = pd.concat([df_numeric, dummies_df], axis=1)
    else:
        df_final = df
        app.logger.info("No categorical columns found for one-hot encoding")

    app.logger.info(f"Feature engineering complete. Dataset shape: {df_final.shape}")
    return df_final


# PHASE 3: MODEL DEVELOPMENT - K-NEAREST NEIGHBORS APPROACH
def prepare_data_for_knn(df):
    """Prepare data for K-Nearest Neighbors model."""
    if df is None:
        app.logger.error("No data available for KNN preparation")
        return None, None, None

    app.logger.info("Preparing data for KNN model...")
    global original_df
    original_df = df.copy()

    # Select available features for similarity calculation
    base_features = ['Age', 'Height', 'Weight', 'BMI', 'Health_Risk_Score']
    optional_features = ['Sex', 'Hypertension', 'Diabetes']

    # Filter to only include features that exist in the dataframe
    numerical_features = [f for f in base_features + optional_features if f in df.columns]
    app.logger.info(f"Using numerical features: {', '.join(numerical_features)}")

    # Get dummy columns for categorical variables
    dummy_prefixes = ['BMI Level_', 'Fitness Goal_', 'Fitness Type_', 'Age_Group_']
    dummy_features = [col for col in df.columns if
                      any(col.startswith(prefix) for prefix in dummy_prefixes)]

    if dummy_features:
        app.logger.info(f"Using {len(dummy_features)} one-hot encoded features")

    features = numerical_features + dummy_features

    # Make sure all features exist in the dataframe
    features = [f for f in features if f in df.columns]

    if not features:
        app.logger.error("Error: No valid features found for KNN model")
        return None, None, None

    # Extract features
    X = df[features].copy()

    # Fill any remaining NaN values with median for numeric columns
    for col in X.select_dtypes(include=['number']).columns:
        X[col] = X[col].fillna(X[col].median())

    # Fill any remaining NaN values with 0 for dummy columns
    for col in dummy_features:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    app.logger.info(f"Data prepared successfully with {X.shape[1]} features")

    return X_scaled, features, scaler


def train_knn_model(X_scaled, n_neighbors=3):
    """Train a K-Nearest Neighbors model."""
    if X_scaled is None:
        app.logger.error("No scaled data available for KNN model training")
        return None

    app.logger.info(f"Training KNN model with {n_neighbors} neighbors...")
    try:
        # Train KNN model
        knn_model = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        knn_model.fit(X_scaled)
        app.logger.info("KNN model trained successfully")
        return knn_model
    except Exception as e:
        app.logger.error(f"Error training KNN model: {str(e)}")
        return None


# Function to preprocess new user data for KNN
def preprocess_user_for_knn(user_data, features, scaler):
    """Process user input data to make it compatible with the KNN model."""
    if not user_data or not features or not scaler:
        app.logger.error("Missing data for user preprocessing")
        return None

    app.logger.info("Processing user profile for recommendation system...")
    user_df = pd.DataFrame([user_data])

    # Calculate BMI if not present
    if 'BMI' not in user_df.columns and 'Height' in user_df.columns and 'Weight' in user_df.columns:
        user_df['BMI'] = user_df['Weight'] / ((user_df['Height'] / 100) ** 2)

    # Add BMI Level if needed
    if 'BMI Level' not in user_df.columns and 'BMI' in user_df.columns:
        user_df['BMI Level'] = pd.cut(
            user_df['BMI'],
            bins=[0, 18.5, 24.9, 29.9, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )

    # Create age groups if needed
    if 'Age_Group' not in user_df.columns and 'Age' in user_df.columns:
        user_df['Age_Group'] = pd.cut(
            user_df['Age'],
            bins=[0, 18, 35, 50, 65, 100],
            labels=['Teen', 'Young_Adult', 'Adult', 'Middle_Age', 'Senior']
        )

    # Initialize health risk score
    user_df['Health_Risk_Score'] = 0
    risk_score = user_df['Health_Risk_Score'].copy()

    # Update health risk based on available data
    if 'Hypertension' in user_df.columns:
        risk_score = np.where(user_df['Hypertension'] == 1, risk_score + 1, risk_score)
    if 'Diabetes' in user_df.columns:
        risk_score = np.where(user_df['Diabetes'] == 1, risk_score + 1, risk_score)
    if 'BMI' in user_df.columns:
        risk_score = np.where(user_df['BMI'] < 18.5, risk_score + 1, risk_score)
        risk_score = np.where(user_df['BMI'] > 30, risk_score + 1, risk_score)
        risk_score = np.where(user_df['BMI'] > 35, risk_score + 1, risk_score)

    user_df['Health_Risk_Score'] = risk_score

    # One-hot encode categorical columns
    for col in ['BMI Level', 'Fitness Goal', 'Fitness Type', 'Age_Group']:
        if col in user_df.columns:
            dummies = pd.get_dummies(user_df[col], prefix=col)
            user_df = pd.concat([user_df, dummies], axis=1)
            user_df = user_df.drop(col, axis=1)

    # Create feature vector with same columns as training data
    feature_vector = pd.DataFrame(columns=features)

    # Fill in available features
    for feature in features:
        if feature in user_df.columns:
            feature_vector[feature] = user_df[feature]
        else:
            feature_vector[feature] = 0  # Default value for missing features

    # Scale the feature vector
    try:
        feature_vector_scaled = scaler.transform(feature_vector)
        app.logger.info("User profile successfully processed")
        return feature_vector_scaled
    except Exception as e:
        app.logger.error(f"Error processing user profile: {str(e)}")
        return None


# Generate recommendations using KNN
def generate_knn_recommendations(knn_model, user_data, features, scaler, n_recommendations=3):
    """Generate fitness recommendations based on similar profiles."""
    if knn_model is None:
        app.logger.error("KNN model is not initialized")
        return []
    
    if user_data is None:
        app.logger.error("User data is empty or invalid")
        return []

    if features is None or scaler is None:
        app.logger.error("Missing model components (features or scaler)")
        return []

    app.logger.info("Finding similar profiles for recommendations...")
    app.logger.info(f"User data: {user_data}")
    
    user_vector = preprocess_user_for_knn(user_data, features, scaler)

    if user_vector is None:
        app.logger.error("Failed to preprocess user data")
        return []

    try:
        # Find nearest neighbors
        distances, indices = knn_model.kneighbors(user_vector)

        # Get recommendations from nearest neighbors
        recommendations = []
        for i in range(min(len(indices[0]), n_recommendations)):
            idx = indices[0][i]
            profile = original_df.iloc[idx]

            # Create a recommendation dictionary with available fields
            recommendation = {
                'Similarity': 1 - distances[0][i],  # Convert distance to similarity score
                'Profile ID': idx
            }

            # Add available fields from the profile
            for field in ['Exercises', 'Equipment', 'Diet (Vegetable)',
                          'Diet (protein intake)', 'Diet (Juice)', 'Recommendation']:
                if field in profile and pd.notna(profile[field]):
                    recommendation[field] = profile[field]
                else:
                    recommendation[field] = "No recommendation available"

            recommendations.append(recommendation)

        app.logger.info(f"Found {len(recommendations)} similar profiles")
        return recommendations
    except Exception as e:
        app.logger.error(f"Error generating recommendations: {str(e)}")
        return []


# Generate personalized recommendations by blending multiple neighbors
def generate_personalized_recommendation(recommendations, user_data):
    """Create a personalized recommendation based on similar profiles and user data."""
    if not recommendations:
        return {}

    app.logger.info("Creating personalized fitness plan...")
    top_rec = recommendations[0]

    # Create personalized recommendation with available fields
    personalized = {}
    for field in ['Exercises', 'Equipment', 'Diet (Vegetable)',
                  'Diet (protein intake)', 'Diet (Juice)', 'Recommendation']:
        if field in top_rec:
            personalized[field] = top_rec[field]
        else:
            personalized[field] = "No recommendation available"

    # Add health-specific modifications
    if user_data.get('Hypertension', 0) == 1:
        personalized['Diet (Vegetable)'] += " (Low sodium options recommended)"
        personalized['Recommendation'] += " Monitor blood pressure regularly during exercise."

    if user_data.get('Diabetes', 0) == 1:
        personalized['Diet (Juice)'] += " (Sugar-free or diluted options only)"
        personalized['Recommendation'] += " Check blood sugar before and after workouts."

    # Customize based on fitness goal
    goal = user_data.get('Fitness Goal', '')
    if goal:
        if 'Weight Loss' in goal:
            personalized['Exercises'] += " Focus on higher repetitions with moderate weight."
        elif 'Weight Gain' in goal:
            personalized['Exercises'] += " Focus on progressive overload with heavier weights."
        elif 'Muscular Fitness' in goal:
            personalized['Exercises'] += " Focus on compound movements and progressive resistance training."

    # Customize based on age
    age = user_data.get('Age', None)
    if age is not None:
        if age > 60:
            personalized['Equipment'] += " Consider using equipment with better joint support."
        elif age < 18:
            personalized['Equipment'] += " Start with bodyweight exercises before progressing to weights."

    app.logger.info("Personalized plan created successfully")
    return personalized


# Function to explain recommendation with more personalization
def explain_personalized_recommendation(recommendation, user_data):
    """Create a personalized explanation of the recommendation."""
    if not recommendation or not user_data:
        return "Unable to generate explanation with insufficient data."

    explanation = f"Based on your unique profile, we've created a personalized recommendation:"

    # Explain based on key metrics
    bmi = user_data.get('BMI', None)
    if bmi is not None:
        if bmi < 18.5:
            explanation += f"<p><strong>BMI Status:</strong> Your BMI of {bmi:.1f} indicates you're underweight. This plan focuses on building strength and healthy weight gain.</p>"
        elif bmi >= 25 and bmi < 30:
            explanation += f"<p><strong>BMI Status:</strong> Your BMI of {bmi:.1f} indicates you're overweight. This plan balances cardio and strength training for optimal weight management.</p>"
        elif bmi >= 30:
            explanation += f"<p><strong>BMI Status:</strong> Your BMI of {bmi:.1f} indicates obesity. This plan prioritizes joint-friendly exercises and nutritional guidance.</p>"
        else:
            explanation += f"<p><strong>BMI Status:</strong> Your BMI of {bmi:.1f} is in the normal range. This plan focuses on maintaining your healthy weight while improving fitness.</p>"

    # Explain based on health conditions
    health_conditions = []
    if user_data.get('Hypertension', 0) == 1:
        health_conditions.append("hypertension")
    if user_data.get('Diabetes', 0) == 1:
        health_conditions.append("diabetes")

    if health_conditions:
        explanation += f"<p><strong>Health Considerations:</strong> Your plan is adjusted for {', '.join(health_conditions)}, with appropriate intensity levels and dietary recommendations.</p>"

    # Explain based on age
    age = user_data.get('Age', None)
    if age is not None:
        if age < 18:
            explanation += f"<p><strong>Age Consideration:</strong> At {age} years old, this plan focuses on fundamentals, proper form, and developing healthy habits.</p>"
        elif age > 65:
            explanation += f"<p><strong>Age Consideration:</strong> At {age} years old, this plan emphasizes joint health, flexibility, and appropriate intensity levels.</p>"
        else:
            explanation += f"<p><strong>Age Consideration:</strong> At {age} years old, this plan balances challenge and sustainability for long-term health.</p>"

    # Explain based on fitness goal
    fitness_goal = user_data.get('Fitness Goal', None)
    if fitness_goal:
        explanation += f"<p><strong>Fitness Goal:</strong> Your goal of '{fitness_goal}' is the foundation of this recommendation, with specific exercise patterns and nutritional guidance to support it.</p>"

    return explanation


# Initialize the model when app starts
def initialize_model():
    global knn_model, features, scaler, original_df

    # Define possible dataset locations
    possible_locations = [
        os.path.join("static", "dataset", "gym_recommendation.csv"),
        os.path.join("static", "dataset"),
        os.path.join("dataset", "gym_recommendation.csv"),
        os.path.join("dataset"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dataset", "gym_recommendation.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "dataset"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "dataset")
    ]
    
    # Search for CSV files in possible locations
    file_path = find_dataset(possible_locations)
    
    if not file_path:
        app.logger.error("No CSV dataset found in any of the expected locations.")
        return False

    # Log the file that will be used
    app.logger.info(f"Loading dataset from: {file_path}")

    # Load the data
    df = load_and_explore_data(file_path)
    if df is None:
        app.logger.error("Failed to load data.")
        return False

    # Feature engineering
    df_processed = feature_engineering(df)
    if df_processed is None:
        app.logger.error("Feature engineering failed.")
        return False

    # Prepare data for KNN
    X_scaled, features_list, scaler_obj = prepare_data_for_knn(df_processed)
    if X_scaled is None:
        app.logger.error("Data preparation failed.")
        return False

    # Train KNN model
    knn = train_knn_model(X_scaled, n_neighbors=3)
    if knn is None:
        app.logger.error("Model training failed.")
        return False

    # Save model components
    knn_model = knn
    features = features_list
    scaler = scaler_obj

    app.logger.info("Recommendation system successfully initialized!")
    return True

def find_dataset(locations):
    """
    Search for a CSV dataset in multiple possible locations.
    Returns the path to the first valid CSV file found, or None if no CSV is found.
    """
    # First check for specific files
    for location in locations:
        if os.path.isfile(location) and location.endswith('.csv'):
            app.logger.info(f"Found dataset at: {location}")
            return location
    
    # Then check directories for any CSV file
    for location in locations:
        if os.path.isdir(location):
            app.logger.info(f"Searching for CSV files in: {location}")
            csv_files = [os.path.join(location, f) for f in os.listdir(location) 
                        if f.endswith('.csv') and os.path.isfile(os.path.join(location, f))]
            
            if csv_files:
                app.logger.info(f"Found {len(csv_files)} CSV files in {location}")
                # Return the first CSV file found
                return csv_files[0]
    
    # Create directory if none exists
    default_dir = os.path.join("static", "dataset")
    if not os.path.exists(default_dir):
        os.makedirs(default_dir, exist_ok=True)
        app.logger.info(f"Created dataset directory at: {default_dir}")
    
    return None

# Remove the deprecated @app.before_first_request decorator
# Instead, add a function to initialize the model
def try_initialize_model():
    """Initialize model if not already initialized."""
    global model_initialized, knn_model, features, scaler
    if not model_initialized:
        model_initialized = initialize_model()
        if model_initialized:
            app.logger.info("Model initialized successfully")
        else:
            app.logger.error("Failed to initialize model")
    return model_initialized

# Route for the home page
@app.route('/', methods=['GET'])
def index():
    # Try to initialize model when the first page is loaded
    try_initialize_model()
    return render_template('index.html')


# Route for the user input form
@app.route('/form', methods=['GET'])
def form():
    # Ensure model is initialized when form page is loaded
    try_initialize_model()
    return render_template('form.html')


# Route to process form submission and display results
@app.route('/recommend', methods=['POST'])
def recommend():
    global knn_model, features, scaler
    
    # Check if model is initialized
    if knn_model is None or features is None or scaler is None:
        app.logger.error("Model components are not initialized. Trying to initialize now...")
        if not try_initialize_model():
            flash('The recommendation system is not ready. Please try again later.', 'error')
            return redirect(url_for('form'))
    
    if request.method == 'POST':
        # Extract user data from form
        user_data = {}

        # Get gender
        gender = request.form.get('gender')
        if gender in ['Male', 'Female']:
            user_data['Sex'] = 1 if gender == 'Male' else 0

        # Get age
        age = request.form.get('age')
        if age and age.isdigit():
            user_data['Age'] = int(age)

        # Get height - handle decimal values
        height = request.form.get('height')
        if height:
            try:
                height_value = float(height)
                # Automatically convert from meters to cm if height is in meters (less than 3)
                if height_value < 3:
                    height_value = height_value * 100
                    app.logger.info(f"Converted height from meters ({height}) to centimeters ({height_value})")
                user_data['Height'] = height_value
            except ValueError:
                app.logger.warning(f"Invalid height value: {height}")

        # Get weight - handle decimal values
        weight = request.form.get('weight')
        if weight:
            try:
                user_data['Weight'] = float(weight)
            except ValueError:
                app.logger.warning(f"Invalid weight value: {weight}")

        # Calculate BMI if possible
        if 'Height' in user_data and 'Weight' in user_data:
            bmi = user_data['Weight'] / ((user_data['Height'] / 100) ** 2)
            user_data['BMI'] = bmi

            # Set BMI Level
            if bmi < 18.5:
                user_data['BMI Level'] = 'Underweight'
            elif bmi < 25:
                user_data['BMI Level'] = 'Normal'
            elif bmi < 30:
                user_data['BMI Level'] = 'Overweight'
            else:
                user_data['BMI Level'] = 'Obese'

        # Get health conditions
        hypertension = request.form.get('hypertension')
        user_data['Hypertension'] = 1 if hypertension == 'Yes' else 0

        diabetes = request.form.get('diabetes')
        user_data['Diabetes'] = 1 if diabetes == 'Yes' else 0

        # Get fitness goal
        fitness_goal = request.form.get('fitness_goal')
        if fitness_goal:
            user_data['Fitness Goal'] = fitness_goal

        # Check if we have the minimum required data
        if not all(k in user_data for k in ['Age', 'Height', 'Weight']):
            flash('Please provide at least your age, height, and weight for better recommendations.', 'warning')
            return redirect(url_for('form'))

        # Generate recommendations
        recommendations = generate_knn_recommendations(knn_model, user_data, features, scaler)

        if recommendations:
            personalized_rec = generate_personalized_recommendation(recommendations, user_data)
            explanation = explain_personalized_recommendation(personalized_rec, user_data)

            # Convert any numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):  # Check if it's a numpy type
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # For numpy scalar types
                    return obj.item()
                else:
                    return obj

            # Convert all numpy types in data to Python native types
            personalized_rec_safe = convert_numpy_types(personalized_rec)

            # Create properly formatted similar profiles with Python native types
            similar_profiles = []
            for rec in recommendations:
                profile_id = rec['Profile ID']
                similarity = rec['Similarity']

                # Convert numpy types if needed
                if hasattr(profile_id, 'item'):
                    profile_id = profile_id.item()
                if hasattr(similarity, 'item'):
                    similarity = similarity.item()

                # Format similarity as a string with 2 decimal places
                similarity_str = f"{float(similarity):.2f}"

                similar_profiles.append({
                    'id': profile_id,
                    'similarity': similarity_str
                })

            # Store session data
            session['recommendation'] = personalized_rec_safe
            session['explanation'] = explanation
            session['similar_profiles'] = similar_profiles

            return render_template(
                'recommendation.html',
                recommendation=personalized_rec_safe,
                explanation=explanation,
                similar_profiles=similar_profiles,
                user_data=user_data
            )
        else:
            flash('Unable to generate recommendations. Please check your input data.', 'error')
            return redirect(url_for('form'))

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/dataset', exist_ok=True)

    # Initialize the model
    if initialize_model():
        app.run(debug=True)
    else:
        print("Failed to initialize the model. Please check the logs for more information.")