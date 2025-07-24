import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import shap
import matplotlib.pyplot as plt

# --- Load Model and necessary files ---
@st.cache_resource
def load_assets():
    """Loads all necessary assets from disk."""
    try:
        model = joblib.load('lgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        explainer = shap.TreeExplainer(model)  # Use TreeExplainer for LightGBM
        return model, scaler, model_columns, explainer
    except FileNotFoundError:
        st.error("Required model files not found. Please ensure all .pkl and .json files are present.")
        return None, None, None, None

model, scaler, model_columns, explainer = load_assets()

# --- App Title and Description ---
st.title('Obesity Risk Prediction Dashboard 🩺')
st.markdown("This dashboard predicts your risk of obesity using a machine learning model and explains the 'why' behind its prediction.")

# --- User Input Section ---
st.sidebar.header('Enter Your Information')

def user_input_features():
    """Creates sidebar widgets for user input."""
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Age = st.sidebar.slider('Age', 14, 65, 25)
    Height = st.sidebar.slider('Height (m)', 1.40, 2.00, 1.70, 0.01)
    Weight = st.sidebar.slider('Weight (kg)', 39.0, 175.0, 70.0, 0.5)
    
    st.sidebar.subheader("Eating Habits")
    FAVC = st.sidebar.selectbox('Do you eat high-caloric food frequently?', ('yes', 'no'))
    FCVC = st.sidebar.slider('How often do you eat vegetables?', 1, 3, 2, help="1: Never, 2: Sometimes, 3: Always")
    NCP = st.sidebar.slider('Number of main meals per day', 1, 4, 3)
    CAEC = st.sidebar.selectbox('How often do you eat between meals?', ('no', 'Sometimes', 'Frequently', 'Always'))
    CH2O = st.sidebar.slider('How much water do you drink daily?', 1, 3, 2, help="1: < 1L, 2: 1-2L, 3: > 2L")
    CALC = st.sidebar.selectbox('How often do you drink alcohol?', ('no', 'Sometimes', 'Frequently'))
    
    st.sidebar.subheader("Physical Condition & History")
    family_history_with_overweight = st.sidebar.selectbox('Family history with overweight?', ('yes', 'no'))
    FAF = st.sidebar.slider('How often do you do physical activity?', 0, 3, 1, help="0: None, 1: 1-2 days/wk, 2: 2-4 days/wk, 3: 4-5 days/wk")
    TUE = st.sidebar.slider('How much time do you spend on screens?', 0, 2, 1, help="0: 0-2h, 1: 3-5h, 2: > 5h")
    MTRANS = st.sidebar.selectbox('Main transportation used?', ('Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'))
    
    data = {
        'Age': Age, 
        'Height': Height, 
        'Weight': Weight, 
        'FCVC': FCVC, 
        'NCP': NCP, 
        'CH2O': CH2O, 
        'FAF': FAF, 
        'TUE': TUE, 
        'Gender': Gender, 
        'family_history_with_overweight': family_history_with_overweight, 
        'FAVC': FAVC, 
        'CAEC': CAEC, 
        'CALC': CALC, 
        'MTRANS': MTRANS, 
        'SMOKE': 'no', 
        'SCC': 'no'
    }
    return pd.DataFrame(data, index=[0])

def preprocess_input(df):
    """Preprocesses the user input in the exact order as the training pipeline."""
    df_processed = df.copy()

    # Step 1: Create Age Category (BEFORE scaling)
    bins = [0, 18, 35, 55, 100]
    labels = ['Adolescent', 'Young Adult', 'Adult', 'Senior']
    df_processed['Age_Category'] = pd.cut(df_processed['Age'], bins=bins, labels=labels, right=False)
    
    # Step 2: Map categorical variables to numerical (BEFORE scaling)
    caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df_processed['CAEC'] = df_processed['CAEC'].map(caec_mapping)
    df_processed['CALC'] = df_processed['CALC'].map(calc_mapping)

    # Step 3: One-hot encode categorical features (BEFORE scaling)
    df_processed = pd.get_dummies(df_processed, columns=[
        'Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC', 'MTRANS', 'Age_Category'
    ])

    # Step 4: Align with model's expected columns BEFORE scaling
    df_processed = df_processed.reindex(columns=model_columns, fill_value=0)
    
    # Step 5: Scale ALL features (the scaler was trained on the full feature set)
    df_processed_scaled = scaler.transform(df_processed)
    df_processed = pd.DataFrame(df_processed_scaled, columns=model_columns)
    
    return df_processed

if model is not None:
    input_df = user_input_features()
    st.subheader('Prediction and Explanation')

    if st.sidebar.button('Predict My Obesity Risk'):
        try:
            processed_input = preprocess_input(input_df)
            prediction_proba = model.predict_proba(processed_input)
            predicted_class_index = np.argmax(prediction_proba)
            obesity_classes = model.classes_
            predicted_class = obesity_classes[predicted_class_index]
            
            # Map numeric predictions to meaningful labels
            class_mapping = {
                0: 'Insufficient Weight',
                1: 'Normal Weight', 
                2: 'Overweight Level I',
                3: 'Overweight Level II',
                4: 'Obesity Type I',
                5: 'Obesity Type II', 
                6: 'Obesity Type III'
            }
            
            # Get meaningful class name
            predicted_class_name = class_mapping.get(predicted_class, f"Class {predicted_class}")
            
            # Display prediction with explanation
            st.success(f"**Predicted Risk Category:** {predicted_class_name}")
            
            # Add explanation for each category
            explanations = {
                'Insufficient Weight': 'Your BMI and lifestyle factors suggest you may be underweight. Consider consulting a healthcare provider about healthy weight gain strategies.',
                'Normal Weight': 'Great news! Your profile suggests you maintain a healthy weight. Keep up your current lifestyle habits.',
                'Overweight Level I': 'Your profile suggests mild overweight status. Small lifestyle changes could help prevent progression to obesity.',
                'Overweight Level II': 'Your profile indicates moderate overweight status. Consider implementing diet and exercise changes.',
                'Obesity Type I': 'Your profile suggests Class I obesity (BMI 30-35). This increases health risks - consider consulting a healthcare provider.',
                'Obesity Type II': 'Your profile suggests Class II obesity (BMI 35-40). This significantly increases health risks - medical consultation recommended.',
                'Obesity Type III': 'Your profile suggests Class III obesity (BMI >40). This poses serious health risks - immediate medical consultation strongly recommended.'
            }
            
            explanation = explanations.get(predicted_class_name, "Please consult with a healthcare provider for personalized advice.")
            st.info(f"**What this means:** {explanation}")
            
            # Show model certainty in a more user-friendly way
            confidence = prediction_proba[0][predicted_class_index]
            if confidence > 0.8:
                certainty_text = "The model is very confident in this prediction."
            elif confidence > 0.6:
                certainty_text = "The model is moderately confident in this prediction."
            else:
                certainty_text = "The model has lower confidence in this prediction - consider this as one of several possible outcomes."
            
            with st.expander("Model Certainty Details"):
                st.write(f"**Model Certainty:** {certainty_text}")
                st.write(f"Technical confidence score: {confidence:.1%}")
                st.write("*Note: High confidence doesn't mean the prediction is definitely correct - it means the model's training data strongly supports this classification based on your inputs.*")

            # SHAP Explanation with user-friendly feature names
            st.write("### Why did the model make this prediction?")
            st.markdown("This chart shows how each of your inputs contributed to the final prediction. **Red bars** pushed toward higher obesity risk, while **blue bars** pushed toward lower risk.")

            # Create user-friendly feature name mapping
            feature_name_mapping = {
                'Age': 'Age',
                'Height': 'Height', 
                'Weight': 'Weight',
                'FCVC': 'Vegetable Consumption',
                'NCP': 'Number of Main Meals',
                'CH2O': 'Daily Water Intake',
                'FAF': 'Physical Activity Frequency',
                'TUE': 'Screen Time',
                'CAEC': 'Snacking Between Meals',
                'CALC': 'Alcohol Consumption',
                'Gender_Male': 'Being Male',
                'Gender_Female': 'Being Female',
                'family_history_with_overweight_yes': 'Family History of Overweight',
                'family_history_with_overweight_no': 'No Family History of Overweight',
                'FAVC_yes': 'Frequent High-Calorie Food Consumption',
                'FAVC_no': 'Avoiding High-Calorie Foods',
                'SMOKE_yes': 'Smoking',
                'SMOKE_no': 'Not Smoking',
                'SCC_yes': 'Calorie Monitoring',
                'SCC_no': 'Not Monitoring Calories',
                'MTRANS_Automobile': 'Using Car for Transportation',
                'MTRANS_Bike': 'Using Bike for Transportation',
                'MTRANS_Motorbike': 'Using Motorbike for Transportation',
                'MTRANS_Public_Transportation': 'Using Public Transportation',
                'MTRANS_Walking': 'Walking for Transportation',
                'Age_Category_Adolescent': 'Being an Adolescent',
                'Age_Category_Young Adult': 'Being a Young Adult',
                'Age_Category_Adult': 'Being an Adult',
                'Age_Category_Senior': 'Being a Senior'
            }
            
            def get_friendly_name(feature_name):
                return feature_name_mapping.get(feature_name, feature_name.replace('_', ' ').title())

            # Calculate SHAP values
            shap_values = explainer.shap_values(processed_input)
            
            # Handle different SHAP value structures
            if isinstance(shap_values, list):
                # Multi-class case: shap_values is a list of arrays, one for each class
                if len(shap_values) > predicted_class_index:
                    current_shap_values = shap_values[predicted_class_index][0]  # Get first sample
                    base_value = explainer.expected_value[predicted_class_index]
                else:
                    current_shap_values = shap_values[0][0]  # Fallback to first class
                    base_value = explainer.expected_value[0] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
            else:
                # Single array case
                if len(shap_values.shape) == 3:  # Shape: (samples, features, classes)
                    current_shap_values = shap_values[0, :, predicted_class_index]
                elif len(shap_values.shape) == 2:  # Shape: (samples, features)
                    current_shap_values = shap_values[0]
                else:
                    current_shap_values = shap_values
                base_value = explainer.expected_value if isinstance(explainer.expected_value, (int, float)) else explainer.expected_value[0]
            
            # Create SHAP waterfall plot
            fig, ax = plt.subplots(figsize=(12, 8))
            try:
                # Create user-friendly feature names for SHAP
                friendly_feature_names = [get_friendly_name(name) for name in processed_input.columns.tolist()]
                
                shap.plots.waterfall(
                    shap.Explanation(
                        values=current_shap_values, 
                        base_values=base_value, 
                        data=processed_input.iloc[0].values,
                        feature_names=friendly_feature_names
                    ), 
                    show=False
                )
                st.pyplot(fig, bbox_inches='tight')
            except Exception as e:
                st.warning(f"Could not generate SHAP waterfall plot, showing alternative visualization.")
                # Fallback: show feature importance bar chart with friendly names
                feature_names = processed_input.columns.tolist()
                shap_values_list = current_shap_values.tolist() if hasattr(current_shap_values, 'tolist') else list(current_shap_values)
                
                # Create DataFrame for visualization
                shap_df_viz = pd.DataFrame({
                    'feature': feature_names,
                    'friendly_name': [get_friendly_name(name) for name in feature_names],
                    'shap_value': shap_values_list
                })
                
                # Sort by absolute value and take top 10
                shap_df_viz['abs_shap'] = shap_df_viz['shap_value'].abs()
                shap_df_viz = shap_df_viz.sort_values('abs_shap', ascending=True).tail(10)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                colors = ['#d62728' if x > 0 else '#1f77b4' for x in shap_df_viz['shap_value']]  # Red for positive, blue for negative
                bars = ax.barh(range(len(shap_df_viz)), shap_df_viz['shap_value'], color=colors, alpha=0.7)
                ax.set_yticks(range(len(shap_df_viz)))
                ax.set_yticklabels(shap_df_viz['friendly_name'], fontsize=10)
                ax.set_xlabel('Impact on Prediction (SHAP Value)', fontsize=12)
                ax.set_title('Top 10 Factors Influencing Your Obesity Risk Prediction', fontsize=14)
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, shap_df_viz['shap_value'])):
                    if value > 0:
                        ax.text(value + 0.01, i, f'+{value:.3f}', va='center', fontsize=9)
                    else:
                        ax.text(value - 0.01, i, f'{value:.3f}', va='center', ha='right', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig, bbox_inches='tight')
            
            plt.close()

            # Data-driven recommendations
            st.write("### Data-Driven Recommendations")
            st.markdown("These recommendations are based on the factors that most increased your predicted risk:")

            # Get feature contributions with friendly names
            feature_names = processed_input.columns.tolist()
            shap_values_list = current_shap_values.tolist() if hasattr(current_shap_values, 'tolist') else list(current_shap_values)
            
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'friendly_name': [get_friendly_name(name) for name in feature_names],
                'shap_value': shap_values_list
            }).sort_values(by='shap_value', ascending=False)

            # Get top risk-increasing factors
            top_risk_factors = shap_df[shap_df['shap_value'] > 0].head(3)
            
            if len(top_risk_factors) == 0:
                st.write("🎉 Your profile shows no major risk-increasing factors. Keep up the healthy habits!")
            else:
                recommendations = []
                
                for _, row in top_risk_factors.iterrows():
                    original_feature = row['feature']
                    friendly_name = row['friendly_name']
                    original_input = input_df.iloc[0]
                    
                    # Generate specific recommendations based on the risk factors
                    if 'Weight' in original_feature:
                        recommendations.append(f"💡 **Weight Management**: Your weight was the strongest factor increasing obesity risk. Consider consulting a healthcare provider about safe weight management strategies.")
                    elif 'FAVC_yes' in original_feature:
                        recommendations.append(f"💡 **Reduce High-Calorie Foods**: Limiting frequent consumption of high-calorie foods could significantly reduce your risk.")
                    elif 'FAF' in original_feature and original_input['FAF'] < 2:
                        recommendations.append(f"💡 **Increase Physical Activity**: Your current activity level ({original_input['FAF']}/3) was a risk factor. Try to exercise more frequently.")
                    elif 'family_history_with_overweight_yes' in original_feature:
                        recommendations.append(f"💡 **Family History Awareness**: While you can't change genetics, maintaining healthy lifestyle habits is especially important given your family history.")
                    elif 'CAEC' in original_feature and original_input['CAEC'] in ['Frequently', 'Always']:
                        recommendations.append(f"💡 **Mindful Snacking**: Reducing snacking between meals could help lower your obesity risk.")
                    elif 'TUE' in original_feature and original_input['TUE'] >= 1:
                        recommendations.append(f"💡 **Reduce Screen Time**: Your screen time ({original_input['TUE']}/2) was contributing to risk. Try to balance with more physical activities.")
                    elif 'CH2O' in original_feature and original_input['CH2O'] < 2:
                        recommendations.append(f"💡 **Increase Water Intake**: Drinking more water daily could support healthy weight management.")
                    elif 'MTRANS_Automobile' in original_feature:
                        recommendations.append(f"💡 **Active Transportation**: Consider walking, biking, or using public transport more often to increase daily activity.")
                    elif 'NCP' in original_feature:
                        if original_input['NCP'] < 3:
                            recommendations.append(f"💡 **Regular Meals**: Eating more regular meals might help with weight management.")
                        elif original_input['NCP'] > 3:
                            recommendations.append(f"💡 **Meal Frequency**: Consider reducing to 3 main meals per day instead of {original_input['NCP']}.")
                    else:
                        recommendations.append(f"💡 **Monitor {friendly_name}**: This factor contributed to your risk prediction.")
                
                # Remove duplicates and display top recommendations
                recommendations = list(dict.fromkeys(recommendations))  # Remove duplicates while preserving order
                for rec in recommendations[:3]:  # Show top 3 recommendations
                    st.info(rec)
                    
                # Add general disclaimer
                st.warning("⚠️ **Important**: These are general recommendations based on the model's analysis. Always consult with a healthcare provider before making significant lifestyle changes.")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please check that all required model files are present and correctly formatted.")
    else:
        st.info('Fill in your details in the sidebar and click "Predict" to see your results.')
else:
    st.error("Could not load the required model files. Please ensure all .pkl and .json files are in the correct directory.")