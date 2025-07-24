import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from datetime import datetime

random_state = 42  # or None

def encode_ordinal(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """Encode ordinal categorical features in the DataFrame
    :param df: DataFrame containing the dataset
    :return: DataFrame with ordinal features encoded
    """
    caec_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_mapping = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3} 

    df['CAEC'] = df['CAEC'].map(caec_mapping)
    df['CALC'] = df['CALC'].map(calc_mapping)

    return df


def categoization_age(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """Categorize age groups in the DataFrame

    :param df: DataFrame containing the dataset
    :return: DataFrame with age categories added
    """
    bins = [0, 18, 35, 55, df['Age'].max()]
    labels = ['Adolescent', 'Young Adult', 'Adult', 'Senior']
    df['Age_Category'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    return df


def encode_target(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """Label Encoding for the target variable `NObeyesdad`

    :param df: DataFrame containing the dataset
    :return: DataFrame with the target variable encoded
    """
    le = LabelEncoder()
    df['NObeyesdad'] = le.fit_transform(df['NObeyesdad'])
    return df


def encode_categorical_features(
        df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features in the DataFrame

    :param df: DataFrame containing the dataset
    :return: DataFrame with categorical features encoded
    """
    binary_features = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for feature in binary_features:
        df[feature] = df[feature].map({'yes': 1, 'no': 0})
    return df


def  one_hot_encode_features(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """One-Hot Encode remaining nominal features
    
    Exclude 'CAEC' and 'CALC' as they are now ordinally encoded
    :param df: DataFrame containing the dataset
    :return: DataFrame with one-hot encoded features
    """
    categorical_features = ['Gender', 'MTRANS', 'Age_Category']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    return df


def drop_features(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """Drop original columns that have been engineered or replaced
    :param df: DataFrame containing the dataset
    :return: DataFrame with specified features dropped
    """
    df.drop(['Age'], axis=1, inplace=True)
    return df


def split_data(
        df: pd.DataFrame
        ) -> tuple:
    """Split Data into Training and Testing Sets
    :param df: DataFrame containing the dataset
    :return: Tuple containing features (X) and target (y)
    """
    # Split Data into Features (X) and Target (y) 
    X = df.drop('NObeyesdad', axis=1)
    y = df['NObeyesdad']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
        )

    return X_train, X_test, y_train, y_test


def scale_data(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame
        ) -> tuple:
    """Scale the features in the training and testing sets
    
    :param X_train: Training pd.DataFrame
    :param X_test: Testing pd.DataFrame
    :return: Tuple containing the scaled training and testing sets
    """
    # 7. Apply Scaling

    # MinMaxScaler 
    scaler = MinMaxScaler(feature_range=(1, 5))

    print(f"\nUsing scaler: {type(scaler).__name__}")

    # Identify numerical features to scale. We now include 'BMI'.
    numerical_features = ['Weight', 'Height', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'CAEC', 'CALC']

    # Create copies to avoid changing the original baseline dataframes
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # Fit the scaler ONLY on the training data
    scaler.fit(X_train_scaled[numerical_features])

    # Transform both the training and testing data
    X_train_scaled[numerical_features] = scaler.transform(X_train_scaled[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test_scaled[numerical_features])

    return X_train_scaled, X_test_scaled



def main():
    DATA_PATH = './data/'

    # Create a Unique Directory for Output Files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(DATA_PATH, 'preprocessed', f"preprocessed_data_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory for output files: '{output_dir}/'")

    # Load Data and Perform Initial Preprocessing 
    input_csv_path = os.path.join(DATA_PATH, 'raw', 'ObesityDataSet_raw_and_data_sinthetic.csv')


    try:
        df = pd.read_csv(input_csv_path)
        print(f"Dataset loaded successfully from: {input_csv_path}")
    except FileNotFoundError:
        print(f"ERROR: The file was not found at '{input_csv_path}'")
        exit()

    df_processed = df.copy()
    df_processed = encode_ordinal(df_processed)
    df_processed = categoization_age(df_processed)
    df_processed = encode_target(df_processed)
    df_processed = encode_categorical_features(df_processed)
    df_processed = one_hot_encode_features(df_processed)
    df_processed = drop_features(df_processed)

    print("\nInitial encoding and feature engineering complete.")
    print("Preprocessed data shape:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df_processed)
    print(f"\nData split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # Save the Baseline (Unscaled) Datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train_unscaled.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test_unscaled.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train_unscaled.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test_unscaled.csv'), index=False)
    print("\nSaved unscaled training and testing sets.")

    # Apply Scaling
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    print("Scaling applied successfully.")


    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train_scaled.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test_scaled.csv'), index=False)
    print("Saved final scaled training and testing sets.")
    print(f"\nAll files saved in: {output_dir}")

if __name__ == "__main__":
    main()