import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Load the CSV file into a DataFrame
csvfile = pd.read_csv(file_path)

def column_type_finder(csvfile):
    def preprocess_numerical_columns(df):
        def convert_value(value):
            if isinstance(value, str):
                value = value.replace('$', '').replace(',', '').replace('₹', '')
                if '%' in value:
                    value = value.replace('%', '')
                    try:
                        value = float(value) / 100
                    except ValueError:
                        return value
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        return value
            return value

        df = df.applymap(convert_value)
        return df

    def convert_to_numerical(df):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        return df

    def is_numerical_series(series):
        if pd.api.types.is_numeric_dtype(series):
            return True
        numeric_count = series.dropna().apply(lambda x: isinstance(x, (int, float))).sum()
        return numeric_count / len(series.dropna()) > 0.8

    def is_categorical_series(series):
        unique_values = series.nunique()
        if unique_values == 2 or unique_values < len(series) * 0.05:
            return True
        return False

    def exclude_empty_columns(df):
        return df.dropna(axis=1, how='all')

    csvfile = exclude_empty_columns(csvfile)
    
    csvfile = preprocess_numerical_columns(csvfile)
    csvfile = convert_to_numerical(csvfile)
    
    def identify_id_columns(data):
        id_cols = []
        for column in data.columns:
            unique_values = data[column].nunique()
            total_values = len(data[column])
            if unique_values == total_values and not pd.api.types.is_float_dtype(data[column]):
                id_cols.append(column)
            elif any(keyword in column.lower() for keyword in ['_id', ' id', 'id ', 'id_', 'ID']):
                id_cols.append(column)
        return id_cols

    def select_id_columns(data, id_cols):
        unique_id_columns = []
        numerical_id_columns = []
        categorical_id_columns = []
        for column in id_cols:
            unique_values = data[column].nunique()
            total_values = len(data[column])
            if unique_values == total_values:
                if pd.api.types.is_float_dtype(data[column]):
                    continue
                unique_id_columns.append(column)
            else:
                if pd.api.types.is_numeric_dtype(data[column]):
                    numerical_id_columns.append(column)
                else:
                    categorical_id_columns.append(column)
        return unique_id_columns, numerical_id_columns, categorical_id_columns

    identified_id_columns = identify_id_columns(csvfile)
    unique_id_columns, numerical_id_columns, categorical_id_columns = select_id_columns(csvfile, identified_id_columns)

    numerical_columns = set()
    categorical_columns = set()
    text_columns = set()
    date_time_columns = set()
    
    for column in csvfile.columns:
        if column in unique_id_columns:
            continue
        if is_numerical_series(csvfile[column]):
            numerical_columns.add(column)
        elif is_categorical_series(csvfile[column]):
            categorical_columns.add(column)
        else:
            text_columns.add(column)

    name_related_keywords = ['name', 'firstname', 'lastname', 'surname']
    def is_name_related(column_name):
        return any(keyword in column_name.lower() for keyword in name_related_keywords)

    for column in csvfile.columns:
        if column in unique_id_columns:
            continue
        if is_name_related(column):
            while True:
                user_input = input(f"Should the Column '{column}' be considered as categorical(1) or text(0)? ")
                if user_input in ['1', '0']:
                    break
                print("Invalid input. Please enter 1 for categorical or 0 for text.")
            if user_input == '1':
                categorical_columns.add(column)
                text_columns.discard(column)
            elif user_input == '0':
                text_columns.add(column)
                categorical_columns.discard(column)

    def date_time_column_finder(dataframe):
        date_time_columns = []
        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':
                try:
                    temp = pd.to_datetime(dataframe[column], errors='coerce')
                    if temp.notna().any():
                        if any(x in column.lower() for x in ['date', 'time', 'timestamp']):
                            date_time_columns.append(column)
                except Exception:
                    continue
        return date_time_columns

    date_time_columns = date_time_column_finder(csvfile)

    text_columns.difference_update(date_time_columns)
    unique_id_columns = [col for col in unique_id_columns if col not in date_time_columns]

    def is_binary_encoded(series):
        return series.nunique() == 2

    for column in csvfile.columns:
        if is_binary_encoded(csvfile[column]):
            categorical_columns.add(column)
            if column in numerical_columns:
                numerical_columns.discard(column)

    numerical_columns_list = list(numerical_columns)
    categorical_columns_list = list(categorical_columns)
    text_columns_list = list(text_columns)
    date_time_columns_list = list(date_time_columns)
    unique_id_columns_list = list(unique_id_columns)

    # Filter out numerical columns that contain any ID-related keywords
    id_keywords = ['_id', ' id', 'id ', 'id_', 'ID']
    numerical_columns_without_id = [
        col for col in numerical_columns_list 
        if not any(keyword in col.lower() for keyword in id_keywords)
    ]

    csvfile = csvfile[numerical_columns_list + categorical_columns_list + text_columns_list + date_time_columns_list + unique_id_columns_list]


    return numerical_columns_list, categorical_columns_list, text_columns_list, date_time_columns_list, unique_id_columns_list, numerical_columns_without_id, csvfile

# Example usage
# Assuming `csvfile` is already defined as a pandas DataFrame
numerical_columns, categorical_columns, text_columns, date_time_columns, unique_id_columns, numerical_columns_without_id, csvfile = column_type_finder(csvfile)


def extract_dates_and_times(dataframe):
    extracted_columns = []
    columns_dropped = []

    # Process specified date/time columns
    for column in date_time_columns:
        if column in dataframe.columns:
            # Convert the column to datetime
            dataframe[column] = pd.to_datetime(dataframe[column], errors='coerce')

            # Extract date components
            if dataframe[column].dt.date.notna().any():
                dataframe[f'{column}_year'] = dataframe[column].dt.year
                dataframe[f'{column}_month'] = dataframe[column].dt.month
                dataframe[f'{column}_day'] = dataframe[column].dt.day
                extracted_columns.extend([f'{column}_year', f'{column}_month', f'{column}_day'])

            # Extract time components
            if dataframe[column].dt.time.notna().any():
                has_hour = dataframe[column].apply(lambda x: x.hour if pd.notna(x) else None).notna().any()
                has_minute = dataframe[column].apply(lambda x: x.minute if pd.notna(x) else None).notna().any()
                has_second = dataframe[column].apply(lambda x: x.second if pd.notna(x) else None).notna().any()

                if has_hour:
                    dataframe[f'{column}_hour'] = dataframe[column].apply(lambda x: x.hour if pd.notna(x) else None)
                    extracted_columns.append(f'{column}_hour')
                if has_minute:
                    dataframe[f'{column}_minute'] = dataframe[column].apply(lambda x: x.minute if pd.notna(x) else None)
                    extracted_columns.append(f'{column}_minute')
                if has_second:
                    dataframe[f'{column}_second'] = dataframe[column].apply(lambda x: x.second if pd.notna(x) else None)
                    extracted_columns.append(f'{column}_second')

            # Track columns to be dropped
            columns_dropped.append(column)

    # Drop the original date/time columns
    if columns_dropped:
        dataframe = dataframe.drop(columns=columns_dropped, errors='ignore')
    else:
        pass

    return dataframe, extracted_columns

# Example usage
# Assume 'csvfile' is your DataFrame
csvfile, extracted_columns = extract_dates_and_times(csvfile)


def drop_empty_and_zero_columns(df):
    # Identify columns that are completely empty
    empty_columns = df.columns[df.isna().all()].tolist()
    
    # Identify columns where all values are zero
    zero_columns = df.columns[(df == 0).all()].tolist()
    
    # Drop columns that are completely empty or have all values as zero
    columns_to_drop = empty_columns + zero_columns
    df_dropped = df.drop(columns=columns_to_drop, errors='ignore')
    
    
    return df_dropped

# Assuming 'csvfile' is your DataFrame
csvfile = drop_empty_and_zero_columns(csvfile)

def fill_missing_with_median(df):
    """Replace missing values in numerical columns of a DataFrame with the median of each column
       and print missing values before and after for numerical columns only."""
    
    # Select numerical columns
    numerical_cols = numerical_columns
    
    # Count missing values before replacement in numerical columns
    missing_before = df[numerical_cols].isna().sum()

    
    # Replace missing values in numerical columns with the median of each column
    for col in numerical_cols:
        # Compute the median of the column, ignoring NaNs
        median = df[col].median()
        # Replace NaNs with the median value
        df[col].fillna(median, inplace=True)
    
    # Count missing values after replacement in numerical columns
    missing_after = df[numerical_cols].isna().sum()

    
    return df

# Example usage
# Assume 'csvfile' is your DataFrame
csvfile = fill_missing_with_median(csvfile)


def remove_duplicates(dataframe):

    # Display the number of duplicate rows before removal
    duplicatecount = dataframe.duplicated().sum()
    
    # Remove duplicates
    dataframe_deduplicated = dataframe.drop_duplicates()
    
    # Display the number of duplicate rows after removal
    duplicatecount1 = dataframe_deduplicated.duplicated().sum()
    return dataframe_deduplicated

csvfile = remove_duplicates(csvfile)

def calculate_max_unique_values(num_rows):
    # Adjust the ratio based on the number of rows
    if num_rows == 0:
        return 0
    # Define the ratio: 1 unique values per 70 rows
    ratio = 1 / 70
    max_unique_values = int(num_rows * ratio)
    # Set a minimum value to avoid too small thresholds
    return max(max_unique_values, 10)  # Ensure a minimum threshold of 10

def identify_label_encoded_columns(dataframe):
    num_rows = len(dataframe)
    max_unique_values = calculate_max_unique_values(num_rows)
    
    label_encoded_columns = []
    
    for column in dataframe.columns:
        if pd.api.types.is_integer_dtype(dataframe[column]):
            unique_values = dataframe[column].nunique()
            if unique_values <= max_unique_values:
                label_encoded_columns.append(column)
    
    
    return label_encoded_columns

def replace_outliers_iqr(dataframe, numerical_columns, label_encoded_columns):
    # Initialize dictionaries to store outlier counts
    outliers_count_before = {}
    outliers_count_after = {}
    
    # Calculate outliers before replacement
    for column in numerical_columns:
        if column in label_encoded_columns:
            # Skip label encoded columns
            continue
        
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 5 * IQR
        upper_bound = Q3 + 5 * IQR
        
        # Identify outliers before replacement
        outliers_before = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outliers_count_before[column] = outliers_before.shape[0]
        
        # Replace outliers with the median
        median_value = dataframe[column].median()
        dataframe[column] = dataframe[column].apply(
            lambda x: median_value if pd.notnull(x) and isinstance(x, (int, float)) and (x < lower_bound or x > upper_bound) else x
        )
        
        # Recalculate outliers after replacement
        outliers_after = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outliers_count_after[column] = outliers_after.shape[0]
    
    
    return dataframe

# Identify potential label encoded columns
label_encoded_columns = identify_label_encoded_columns(csvfile)

# Replace outliers while excluding label encoded columns
csvfile = replace_outliers_iqr(csvfile, numerical_columns, label_encoded_columns)

# Display the updated DataFrame
csvfileviz = csvfile.copy()


def encode_categorical(data, categorical_columns):
    def encode_column(column):
        unique_values = column.nunique()
        if unique_values == 2:
            # Label encoding for binary data
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(column)
            encoding_method = "Label encoding (binary)"
            return pd.DataFrame(encoded, columns=[column.name], index=column.index), encoding_method
        elif unique_values < row_count * 0.05:
            # Label encoding for low-cardinality data
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(column)
            encoding_method = "Label encoding (low-cardinality)"
            return pd.DataFrame(encoded, columns=[column.name], index=column.index), encoding_method
        else:
            # Frequency encoding for high-cardinality data
            freq_encoding = column.map(column.value_counts()) / len(column)
            encoding_method = "Frequency encoding"
            return pd.DataFrame(freq_encoding, columns=[column.name], index=column.index), encoding_method

    row_count = len(data)

    # Define high cardinality threshold
    high_cardinality_threshold = row_count * 0.9

    def is_high_cardinality(col):
        return len(col.unique()) > high_cardinality_threshold

    # Identify columns to exclude based on high cardinality
    columns_to_exclude = [col for col in categorical_columns if is_high_cardinality(data[col])]

    # Filter out columns to encode
    columns_to_encode = [col for col in categorical_columns if col not in columns_to_exclude]

    if columns_to_encode:
        encoded_data = pd.DataFrame(index=data.index)
        for column_name in columns_to_encode:
            if column_name in data.columns:
                encoded_column, encoding_method = encode_column(data[column_name])
                encoded_data = pd.concat([encoded_data, encoded_column], axis=1)

        non_categorical_data = data.drop(columns=columns_to_encode, errors='ignore')
        final_data = pd.concat([non_categorical_data, encoded_data], axis=1)
    else:
        final_data = data

    return final_data

# Example usage with a DataFrame named csvfile
csvfile = encode_categorical(csvfile, categorical_columns)

# Check if the count of columns in numerical_columns_without_id is zero
if len(numerical_columns_without_id) == 0:
    csvfile_zero_numc = csvfile.copy()
    

def feature_scaling(data, numerical_columns, text_columns, unique_id_columns):
    
    def is_gaussian_distribution(data):
        p_values = []
        for column in data.columns:
            if data[column].dtype in ['int64', 'int32', 'float64']:
                # Drop NaN values before testing
                stat, p_value = stats.shapiro(data[column].dropna())
                p_values.append(p_value)
        # If any p-value is below 0.05, the data is not normally distributed
        return all(p > 0.05 for p in p_values)
    
    def standardize_data(data):
        scaler = StandardScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    
    def normalize_data(data):
        scaler = MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    # Ensure all numerical columns are included
    all_numerical_columns = list(set(numerical_columns).union(
        data.select_dtypes(include=['int64', 'int32', 'float64']).columns))
    
    # Exclude text columns and unique ID columns
    columns_to_exclude = set(text_columns + unique_id_columns)
    columns_to_exclude = [col for col in columns_to_exclude if col in data.columns]  # Ensure columns exist
    
    # Exclude columns to exclude from scaling
    columns_to_scale = [col for col in all_numerical_columns if col not in columns_to_exclude]


    # Select only the relevant numeric columns for scaling
    numeric_data = data[columns_to_scale]

    if numeric_data.empty:
        print("No numeric columns to scale.")
        scaled_data = pd.DataFrame()
    else:
        # Determine whether to standardize or normalize
        if is_gaussian_distribution(numeric_data):
            scaled_data = standardize_data(numeric_data)
        else:
            scaled_data = normalize_data(numeric_data)

    # Replace the values in the original columns with the scaled data
    data[columns_to_scale] = scaled_data
    # Drop excluded columns, if any
    data = data.drop(columns=columns_to_exclude, errors='ignore')
    return data

# Example usage
# Assuming numerical_columns, text_columns, and unique_id_columns are predefined
csvfile = feature_scaling(csvfile, numerical_columns, text_columns, unique_id_columns)


# Function to calculate VIF
def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["feature"] = dataframe.columns
    vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    return vif_data

# Function to handle multicollinearity based on both VIF and correlation
def handle_multicollinearity(dataframe, target_variable, vif_threshold, corr_threshold):
    dropped_features = []  # List to store dropped features

    # Ensure the target variable is excluded from feature processing
    features = dataframe.drop(columns=[target_variable])

    while True:
        # Step 1: Calculate VIF (exclude target variable)
        vif_data = calculate_vif(features.select_dtypes(include=[np.number]))
        high_vif = vif_data[vif_data["VIF"] > vif_threshold]

        # Step 2: Get correlation matrix (exclude target variable)
        correlation_matrix = features.corr().abs()

        # Step 3: Identify highly correlated features
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_correlation_pairs = [(column, upper_triangle[column].idxmax(), upper_triangle[column].max()) 
                                  for column in upper_triangle.columns 
                                  if upper_triangle[column].max() > corr_threshold]

        # Step 4: Drop the most correlated feature if correlation exceeds threshold
        if high_correlation_pairs:
            feature_to_drop = high_correlation_pairs[0][0]  # Arbitrarily drop the first pair
            features = features.drop(columns=[feature_to_drop])
            dataframe = dataframe.drop(columns=[feature_to_drop])  # Also drop from the original dataframe
            dropped_features.append(feature_to_drop)  # Add to dropped features list
            continue  # Continue loop to re-check correlation and VIF after drop

        # Step 5: Drop feature with highest VIF if above threshold
        if not high_vif.empty:
            feature_to_drop = high_vif['feature'].iloc[0]
            features = features.drop(columns=[feature_to_drop])
            dataframe = dataframe.drop(columns=[feature_to_drop])  # Also drop from the original dataframe
            dropped_features.append(feature_to_drop)  # Add to dropped features list
            continue  # Continue loop to re-check VIF after drop
        
        # Step 6: Break if no features to drop
        break
        
    return dataframe, dropped_features  # Return the DataFrame and the list of dropped features

# Main code execution
vif_threshold = 10   # VIF threshold
corr_threshold = 0.9 # Correlation threshold

# Combine numerical and categorical columns for target selection
all_columns = numerical_columns + categorical_columns

# Step 2: Display all columns and ask the user to select the target variable
for i, col in enumerate(all_columns, 1):
    print(f"{i}. {col}")

# Simulate user input for target variable
target_variable_index = int(input("Please select the variable you want to Predict : ")) - 1
target_variable = all_columns[target_variable_index]


# Store the target variable name in a variable
stored_target_variable = target_variable

# Step  3: Calculate initial VIF (excluding target variable)
initial_vif_data = calculate_vif(csvfile.drop(columns=[target_variable]).select_dtypes(include=[np.number]))

# Step 4: Handle multicollinearity based on VIF and Correlation
csvfile, dropped_columns = handle_multicollinearity(csvfile, target_variable, vif_threshold, corr_threshold)

# Step 5: Recheck final VIF (excluding target variable)
final_vif_data = calculate_vif(csvfile.drop(columns=[target_variable]).select_dtypes(include=[np.number]))


# Step 6: Display the dropped columns
for col in dropped_columns:
    if col in csvfileviz.columns:  # Check if the column exists in csvfileviz
        csvfileviz = csvfileviz.drop(columns=[col])

combined_dict = {}

# Function to initialize combined_dict for a specific column if not already initialized
def initialize_combined_dict(columnName):
    if columnName not in combined_dict:
        combined_dict[columnName] = dict(zip(csvfile[columnName], csvfileviz[columnName]))

# Get the value for a given key
def getValue(key, columnName):
    initialize_combined_dict(columnName)
    return combined_dict[columnName].get(key)

# Get the key for a given value (same as getValue in your original code)
def getKey(key, columnName):
    initialize_combined_dict(columnName)
    return combined_dict[columnName].get(key)

# Get the key from a value, or the nearest key if exact match is not found
def get_key_from_value(val, columnName):
    initialize_combined_dict(columnName)
    
    # Search for the exact match
    for key, value in combined_dict[columnName].items():
        if value == val:
            return key
    
    # If no exact match, return the nearest key
    return getNearestKey(val, columnName)

# Get the nearest key based on the value
def getNearestKey(val, columnName):
    initialize_combined_dict(columnName)
    
    nearestVal = None
    smallest_difference = float('inf')  # Initialize with infinity
    for key, value in combined_dict[columnName].items():
        dif = abs(val - value)
        if dif < smallest_difference:
            smallest_difference = dif
            nearestVal = value
    
    # Return the key associated with the nearest value
    return get_key_from_value(nearestVal, columnName)

# Get key from same file (csvF), searching for an exact match
def getKeyFromSameFile(val, csvF, columnName):
    for i in csvF[columnName]:
        if i == val:
            return i
    
    # If no exact match, return the nearest key from the same file
    return getNearestKeyFromSameFile(val, csvF, columnName)

# Find the nearest key in the same file
def getNearestKeyFromSameFile(val, csvF, columnName):
    nearestVal = None
    smallest_difference = float('inf')  # Initialize with infinity
    for i in csvF[columnName]:
        dif = abs(val - i)
        if dif < smallest_difference:
            smallest_difference = dif
            nearestVal = i
    
    return nearestVal
csvfile.head()

# Function to train different regression models and return their performance
def train_and_evaluate_models(X, y):
    # Set a random state for reproducibility in stochastic models
    random_state = 42

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=random_state),
        'Lasso Regression': Lasso(random_state=random_state),
        'ElasticNet': ElasticNet(random_state=random_state),
        'Bayesian Ridge Regression': BayesianRidge(),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'Random Forest': RandomForestRegressor(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
        'AdaBoost': AdaBoostRegressor(random_state=random_state),
        'Extra Trees Regressor': ExtraTreesRegressor(random_state=random_state),
        'Bagging Regressor': BaggingRegressor(random_state=random_state),
        'Support Vector Regression (SVR)': SVR(),
        'K-Nearest Neighbors (KNN)': KNeighborsRegressor(),
        'Neural Network (MLP)': MLPRegressor(max_iter=500, random_state=random_state)
    }

    model_performance = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)  # Train without scaling
            predictions = model.predict(X_test)  # Predict without scaling
            r2 = r2_score(y_test, predictions)
            accuracy = r2 * 100  # Convert R^2 to percentage
            mse = mean_squared_error(y_test, predictions)
            model_performance[model_name] = {
                'accuracy': accuracy,
                'mse': mse,
                'model_instance': model
            }
        except Exception as e:
            model_performance[model_name] = {
                'accuracy': None,
                'mse': None,
                'error': str(e),
                'model_instance': None
            }

    return model_performance  # No need to return scaler since we're not using it

def collect_user_input(X_viz, target_variable, text_columns, unique_id_columns):
    user_input = {}
    print("Enter the values from the new Obseravtion:")

    for feature in X_viz.columns:
        if feature != target_variable and feature not in text_columns and feature not in unique_id_columns:
            if X_viz[feature].dtype == 'object':  # Categorical feature
                unique_values = X_viz[feature].unique()
                print(f"\nSelect a value for {feature}:")
                for idx, val in enumerate(unique_values):
                    print(f"{idx}: {val}")
                selected_value_idx = int(input(f"Enter the number corresponding to the value for {feature}: "))
                user_input[feature] = get_key_from_value(unique_values[selected_value_idx],feature)
            else:  # Numerical feature
                min_value = X_viz[feature].min()
                max_value = X_viz[feature].max()
                user_input_value = get_key_from_value(float(input(f"Enter a value for {feature} . The Recommended Range is from {min_value} to {max_value} : ")),feature)

                user_input[feature] = user_input_value
    
    return user_input

# Main code execution
# Load your CSV file into a DataFrame (example: csvfile = pd.read_csv('your_file.csv'))
# Define the target variable (example: target_variable = 'your_target_column')

# Assuming csvfile is your DataFrame and target_variable is defined
X = csvfile.drop(columns=[target_variable])
y = csvfile[target_variable]

model_performance = train_and_evaluate_models(X, y)

# Step 3: Initialize variable for the closest model based on closeness to 87.5%
final_closest_model = None
final_closest_model_name = None  # Variable to store the name of the closest model
closest_accuracy = None
smallest_difference = float('inf')  # Initialize with infinity
	
	

# Loop through model performance to find the model closest to 87.5%
for name, data in model_performance.items():
    accuracy = data['accuracy']
    if accuracy is not None:
        difference = abs(accuracy - 87.5)  # Calculate the difference from 87.5%
        if difference < smallest_difference:
            smallest_difference = difference
            final_closest_model = data['model_instance']
            final_closest_model_name = name  # Store the model name
            closest_accuracy = accuracy

# Step 4: Display the closest model to 87.5%
if final_closest_model:
    print(f"The Prediction is {closest_accuracy:.2f} % accurate")
else:
    print("No suitable model found.")

# Prediction logic using final_closest_model
if final_closest_model:
    # Collect user input for prediction
    user_input = collect_user_input(csvfileviz, target_variable, text_columns, unique_id_columns)

    # Prepare input for the prediction model
    input_data = pd.DataFrame([user_input])

    # Ensure input_data has the correct columns in the same order as X
    input_data = input_data.reindex(columns=X.columns, fill_value=0)  # Fill missing columns with a default value (0 or suitable default)

    # Make sure the input has the correct number of features
    if input_data.shape[1] == X.shape[1]:  # Compare feature count

        # Make prediction without scaling
        predicted_value = final_closest_model.predict(input_data)

        print(f"The predicted value for the {target_variable} is {getValue(getKeyFromSameFile(float(predicted_value[0]), csvfile, target_variable), target_variable)}")

    else:
        print("Error: Mismatch in feature count.")
else:
    print("No model was selected for prediction.")
