"""
Obesity Risk Prediction Dashboard
This Streamlit application provides an interactive interface for users to input
their demographic and lifestyle information and receive obesity risk predictions
using a pre-trained LightGBM model.
Features:
- Interactive input form for user data
- Real-time obesity risk prediction
- Visual scale showing obesity levels
- Model confidence and explanation
Usage:
    streamlit run streamlit_app.py
Requirements:
    - streamlit
    - pandas
    - numpy
    - lightgbm
    - joblib
    - plotly
    - sklearn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys

# Add parent directory to path to import preprocessing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Page configuration
st.set_page_config(
    page_title="Obesity Risk Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Obesity level mapping
OBESITY_LEVELS = {
    0: "Insufficient Weight",
    1: "Normal Weight", 
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

OBESITY_COLORS = {
    0: "#3498db",  # Blue
    1: "#2ecc71",  # Green
    2: "#f39c12",  # Orange
    3: "#e67e22",  # Dark Orange
    4: "#e74c3c",  # Red
    5: "#c0392b",  # Dark Red
    6: "#8e44ad"   # Purple
}

@st.cache_data
def load_latest_model():
    """Load the most recent trained model and its metadata"""
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    if not os.path.exists(models_dir):
        st.error(f"Models directory not found: {models_dir}")
        return None, None, None

    # Find the latest model file
    model_files = [f for f in os.listdir(models_dir) if f.startswith('lgbm_model_') and f.endswith('.pkl')]

    if not model_files:
        st.error("No trained models found in the models directory")
        return None, None, None

    latest_model_file = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model_file)

    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    # Load metadata if available
    metadata = None
    metadata_files = [f for f in os.listdir(models_dir) if f.startswith('lgbm_metadata_') and f.endswith('.pkl')]
    if metadata_files:
        latest_metadata_file = sorted(metadata_files)[-1]
        metadata_path = os.path.join(models_dir, latest_metadata_file)
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except:
            metadata = None

    # Load scaler
    scaler = None
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')
    if os.path.exists(data_dir):
        # Find the latest preprocessed data folder
        folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if folders:
            latest_folder = sorted(folders)[-1]
            scaler_path = os.path.join(data_dir, latest_folder, 'scaler.pkl')
            if os.path.exists(scaler_path):
                try:
                    scaler = joblib.load(scaler_path)
                except:
                    scaler = None

    return model, metadata, scaler

def preprocess_user_input(user_data, scaler=None):
    """Preprocess user input to match the training data format"""

    # Create a DataFrame from user input
    df = pd.DataFrame([user_data])

    # Apply the same preprocessing steps as in training

    # 1. Encode ordinal features
    caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

    df['CAEC'] = df['CAEC'].map(caec_mapping)
    df['CALC'] = df['CALC'].map(calc_mapping)

    # 2. Categorize age
    bins = [0, 18, 35, 55, 100]  # Using 100 as max age
    labels = ['Adolescent', 'Young Adult', 'Adult', 'Senior']
    df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    # 3. Encode binary features
    binary_features = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for feature in binary_features:
        df[feature] = df[feature].map({'yes': 1, 'no': 0})

    # 4. One-hot encode categorical features
    categorical_features = ['Gender', 'MTRANS', 'Age_Category']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 5. Drop Age column (replaced by Age_Category)
    df.drop(['Age'], axis=1, inplace=True)

    # 6. Ensure all expected columns are present (from training data)
    expected_columns = [
        'Weight', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC',
        'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC',
        'Gender_Male', 'MTRANS_Bike', 'MTRANS_Motorbike', 'MTRANS_Public_Transportation',
        'MTRANS_Walking', 'Age_Category_Adult', 'Age_Category_Senior', 'Age_Category_Young Adult'
    ]

    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[expected_columns]

    # 7. Apply scaling if scaler is available
    if scaler is not None:
        numerical_features = ['Weight', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC']
        df_scaled = df.copy()
        df_scaled[numerical_features] = scaler.transform(df[numerical_features])
        return df_scaled
    else:
        st.warning("No scaler found - using unscaled data")
        return df

def create_obesity_scale(predicted_level, confidence_scores=None):
    """Create a visual scale showing where the user falls on the obesity spectrum"""

    levels = list(OBESITY_LEVELS.values())
    colors = list(OBESITY_COLORS.values())

    # Create a list of opacities: 1.0 for the predicted level, 0.3 for others
    opacities = [0.3] * len(levels)
    opacities[predicted_level] = 1.0

    fig = go.Figure()

    # Create a single bar trace to prevent duplication
    fig.add_trace(go.Bar(
        x=list(range(len(levels))),
        y=[1] * len(levels),
        text=levels,
        textposition="inside",
        textfont=dict(size=10, color="white"),
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(width=0)
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
        showlegend=False
    ))

    # Add annotation below the predicted level, without an arrow
    fig.add_annotation(
        x=predicted_level,
        y=-0.3,  # Position the text below the bar
        text="<b>YOU ARE HERE</b>",
        showarrow=False,  # Remove the arrow
        font=dict(size=14, color="red", family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=2
    )

    # Update layout to make space for the annotation below
    fig.update_layout(
        title="Your Position on the Obesity Scale",
        title_font_size=20,
        title_x=0.5,
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=250,
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 1.5]),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(7)),
            ticktext=[f"Level {i}" for i in range(7)],
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=50, l=20, r=20)
    )

    return fig

def create_confidence_chart(confidence_scores):
    """Create a chart showing prediction confidence for all classes"""

    levels = list(OBESITY_LEVELS.values())
    colors = list(OBESITY_COLORS.values())

    # Find the predicted class (highest confidence)
    predicted_idx = np.argmax(confidence_scores)

    # CORRECTED PART: Changed '#lightgray' to 'lightgray'
    fig = go.Figure(data=[
        go.Bar(
            x=levels,
            y=confidence_scores,
            marker=dict(
                color=[colors[i] if i == predicted_idx else 'lightgray' for i in range(len(colors))],
                opacity=[1.0 if i == predicted_idx else 0.4 for i in range(len(colors))]
            ),
            text=[f"{score:.1%}" if i == predicted_idx else f"{score:.0%}" for i, score in enumerate(confidence_scores)],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Model Confidence Across All Levels",
        title_font_size=16,
        title_x=0.5,
        xaxis_title="Obesity Levels",
        yaxis_title="Confidence (%)",
        height=400,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickformat='.0%')
    )

    return fig

def main():
    """Main Streamlit application"""

    # Title and description
    st.title("🩺 Obesity Risk Prediction Dashboard 🩺")
    st.markdown("""
    This application uses machine learning to predict your obesity risk level based on your 
    demographic information and lifestyle factors. Please fill in the form below to get your prediction.
    """)

    # Load model
    with st.spinner("Loading model..."):
        model, metadata, scaler = load_latest_model()

    if model is None:
        st.error("Failed to load the trained model. Please ensure you have trained a model first.")
        st.stop()

    # Create input form
    st.header("📝 Enter Your Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", min_value=14, max_value=80, value=25, help="Your age in years")
        height = st.slider("Height (m)", min_value=1.40, max_value=2.10, value=1.70, step=0.01, help="Your height in meters")
        weight = st.slider("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.5, help="Your weight in kilograms")

        st.subheader("Family History")
        family_history = st.selectbox("Family history with overweight?", ["no", "yes"])

    with col2:
        st.subheader("Eating Habits")
        favc = st.selectbox("Do you eat high caloric food frequently?", ["no", "yes"])
        fcvc = st.slider("Frequency of vegetables consumption", min_value=1.0, max_value=3.0, value=2.0, step=0.1, 
                         help="1 = Never, 2 = Sometimes, 3 = Always")
        ncp = st.slider("Number of main meals", min_value=1.0, max_value=4.0, value=3.0, step=0.1)
        caec = st.selectbox("Consumption of food between meals", ["no", "Sometimes", "Frequently", "Always"])
        ch2o = st.slider("Consumption of water daily (L)", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        calc = st.selectbox("Consumption of alcohol", ["no", "Sometimes", "Frequently", "Always"])

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Physical Activity & Lifestyle")
        faf = st.slider("Physical activity frequency (days/week)", min_value=0.0, max_value=3.0, value=1.0, step=0.1)
        tue = st.slider("Time using technology devices (hours/day)", min_value=0.0, max_value=5.0, value=2.0, step=0.1)

    with col4:
        st.subheader("Other Habits")
        smoke = st.selectbox("Do you smoke?", ["no", "yes"])
        scc = st.selectbox("Do you monitor calorie consumption?", ["no", "yes"])
        mtrans = st.selectbox("Transportation used", 
                              ["Automobile", "Walking", "Public_Transportation", "Bike", "Motorbike"])

    # Create user data dictionary
    user_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'CH2O': ch2o,
        'CALC': calc,
        'FAF': faf,
        'TUE': tue,
        'SMOKE': smoke,
        'SCC': scc,
        'MTRANS': mtrans
    }

    # Calculate BMI for display
    bmi = weight / (height ** 2)

    # Display BMI
    st.subheader("📈 Your BMI")
    st.metric("Body Mass Index", f"{bmi:.1f}")

    if bmi < 18.5:
        st.info("BMI Category: Underweight")
    elif 18.5 <= bmi < 25:
        st.success("BMI Category: Normal weight")
    elif 25 <= bmi < 30:
        st.warning("BMI Category: Overweight")
    else:
        st.error("BMI Category: Obese")

    # Prediction button
    if st.button("🔮 Predict Obesity Risk", type="primary"):
        try:
            with st.spinner("Making prediction..."):
                # Preprocess user input
                processed_data = preprocess_user_input(user_data, scaler)

                # Make prediction
                prediction = model.predict(processed_data)[0]
                prediction_proba = model.predict_proba(processed_data)[0]

                # Display results
                st.header("🎯 Your Results")

                # Main prediction with better styling
                predicted_level = OBESITY_LEVELS[prediction]
                confidence = prediction_proba[prediction]

                # Create a prominent result box
                result_color = "green" if prediction <= 1 else "orange" if prediction <= 3 else "red"
                st.markdown(f"""
                <div style="
                    background-color: {result_color}20; 
                    border-left: 5px solid {result_color}; 
                    padding: 20px; 
                    margin: 20px 0;
                    border-radius: 5px;
                ">
                    <h2 style="color: {result_color}; margin: 0;">
                        Your Obesity Level: {predicted_level}
                    </h2>
                    <p style="font-size: 18px; margin: 10px 0 0 0;">
                        Model Confidence: {confidence:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Visual scale with arrow
                scale_fig = create_obesity_scale(prediction, prediction_proba)
                st.plotly_chart(scale_fig, use_container_width=True)

                # Confidence chart (simplified)
                confidence_fig = create_confidence_chart(prediction_proba)
                st.plotly_chart(confidence_fig, use_container_width=True)

                # Recommendations
                st.header("💡 Recommendations")

                if prediction <= 1:
                    st.success("""
                    **Great job!** You're in a healthy weight range. Keep up your current lifestyle:
                    - Maintain your current diet and exercise routine
                    - Continue monitoring your health regularly
                    - Stay hydrated and eat balanced meals
                    """)
                elif prediction <= 3:
                    st.warning("""
                    **Consider lifestyle changes** to improve your health:
                    - Increase physical activity to at least 150 minutes per week
                    - Focus on a balanced diet with more vegetables and fruits
                    - Reduce consumption of high-calorie foods
                    - Monitor portion sizes
                    - Consider consulting with a healthcare professional
                    """)
                else:
                    st.error("""
                    **Important:** You may be at higher risk. Consider taking action:
                    - Consult with a healthcare professional or nutritionist
                    - Develop a structured exercise plan
                    - Focus on significant dietary changes
                    - Consider medical evaluation for underlying conditions
                    - Set realistic weight loss goals with professional guidance
                    """)

                # Feature importance (only for overweight/obese users)
                if prediction > 1 and hasattr(model, 'feature_importances_'):
                    st.header("📊 Key Factors Affecting Your Weight")

                    # Create user-friendly feature names mapping
                    feature_name_mapping = {
                        'Weight': 'Your Current Weight',
                        'Height': 'Your Height',
                        'FCVC': 'Vegetable Consumption',
                        'NCP': 'Number of Main Meals',
                        'CH2O': 'Daily Water Intake',
                        'FAF': 'Physical Activity Frequency',
                        'TUE': 'Technology Use Time',
                        'CAEC': 'Eating Between Meals',
                        'CALC': 'Alcohol Consumption',
                        'family_history_with_overweight': 'Family History of Overweight',
                        'FAVC': 'High Calorie Food Consumption',
                        'SMOKE': 'Smoking Habit',
                        'SCC': 'Calorie Monitoring',
                        'Gender_Male': 'Gender (Male)',
                        'MTRANS_Bike': 'Transportation: Bike',
                        'MTRANS_Motorbike': 'Transportation: Motorbike',
                        'MTRANS_Public_Transportation': 'Transportation: Public Transport',
                        'MTRANS_Walking': 'Transportation: Walking',
                        'Age_Category_Adult': 'Age Group: Adult',
                        'Age_Category_Senior': 'Age Group: Senior',
                        'Age_Category_Young Adult': 'Age Group: Young Adult'
                    }

                    feature_names = processed_data.columns
                    importances = model.feature_importances_
                    user_values = processed_data.iloc[0].values

                    # Create the full feature importance dataframe
                    full_importance_df = pd.DataFrame({
                        'Feature': [feature_name_mapping.get(name, name) for name in feature_names],
                        'Importance': importances,
                        'Your_Value': user_values
                    })

                    # Define and filter out non-actionable features
                    features_to_exclude = ['Your Height', 'Gender (Male)']
                    filtered_df = full_importance_df[~full_importance_df['Feature'].isin(features_to_exclude)]

                    # Select the top 8 from the filtered list to plot
                    importance_df_to_plot = filtered_df.sort_values('Importance', ascending=True).tail(8)

                    # Create horizontal bar chart
                    fig = px.bar(
                        importance_df_to_plot, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Factors Most Influencing Your Weight Category',
                        labels={'Importance': 'Impact on Prediction', 'Feature': ''},
                        color='Importance',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.info("""
                    💡 **Understanding This Chart**: 
                    - Higher bars indicate factors that have more influence on your weight category prediction
                    - Focus on modifiable factors like physical activity, diet, and lifestyle choices
                    - Consider discussing these factors with a healthcare professional
                    """)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.error("Please check your input values and try again.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This prediction is for educational purposes only and should not replace 
    professional medical advice. Please consult with healthcare professionals for medical decisions.
    """)

if __name__ == "__main__":
    main()