import pandas  as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


def read_txt_file(file_path, delimiter):
    try:
        # Read the text file using the provided delimiter
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df
    
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def missing_values_summary(df):
    # Absolute count of missing values
    missing_count = df.isnull().sum()

    # Percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # Combine the two into a DataFrame
    missing_data = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage (%)': missing_percentage
    })

    return missing_data


def univariate_analysis_with_subplots(df):
    # Number of columns in the DataFrame
    num_columns = df.shape[1]
    
    # Determine number of rows and columns for subplots
    num_rows = math.ceil(num_columns / 2)  # Assuming 2 plots per row
    
    # Set up the figure for subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 6 * num_rows))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    for i, column in enumerate(df.columns):
        ax = axes[i]
        
        # Check if the column is numerical
        if pd.api.types.is_numeric_dtype(df[column]):
            # Plot histogram for numerical columns
            sns.histplot(df[column].dropna(), kde=True, bins=10, ax=ax)
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        
        # Check if the column is categorical
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            # Plot bar chart for categorical columns
            sns.countplot(x=df[column], ax=ax)
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            # Rotate x-axis tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Remove any extra subplots if there are empty axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()



def bivariate_multivariate_analysis(df, hub_column):
    # Scatter plot for TotalPremium vs TotalClaims colored by hub column
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df, hue=hub_column, palette='viridis', s=100, alpha=0.7)
    plt.title(f'Scatter Plot of TotalPremium vs TotalClaims by {hub_column}')
    plt.xlabel('TotalPremium')
    plt.ylabel('TotalClaims')
    plt.legend(title=hub_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Correlation matrix for numerical columns
    corr = df[['TotalPremium', 'TotalClaims']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix of TotalPremium and TotalClaims')
    plt.tight_layout()
    plt.show()



def create_crosstab(df, col1, col2):
    if col1 in df.columns and col2 in df.columns:
        crosstab_result = pd.crosstab(df[col1], df[col2])
        return crosstab_result
    else:
        return f"Error: Columns {col1} and/or {col2} not found in DataFrame."


def visualize_correlation_matrix(df, numerical_features):
    # Correlation matrix
    corr = df[numerical_features].corr()

    # Visualize correlation
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix for Numerical Features')
    plt.show()


def plot_premium_vs_claims(df, x='TotalPremium', y='TotalClaims', hue='VehicleType'):
    # Plot using sns.lmplot
    sns.lmplot(x=x, y=y, hue=hue, data=df, height=6, aspect=1.5)

    # Adding title and labels
    plt.title(f'{x} vs {y} by {hue}')
    plt.xlabel(x)
    plt.ylabel(y)

    # Add grid and layout adjustments
    plt.grid(True)
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_correlation_heatmap(df, numerical_features, title='Correlation Heatmap of Numerical Features'):
    # Compute correlation matrix
    corr_matrix = df[numerical_features].corr()

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', linewidths=0.5)

    # Add title and layout adjustments
    plt.title(title)
    plt.tight_layout()

    # Display the plot
    plt.show()


def plot_boxplot(df, x_col, y_col, plot_title):
    plt.figure(figsize=(12, 6)) 
    sns.boxplot(x=x_col, y=y_col, data=df, palette="Set3")
    plt.xticks(rotation=90)
    plt.title(plot_title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.show()
    

def impute_numerical_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    numerical_cols = df.select_dtypes(include=['float64']).columns
    cols_to_impute = [col for col in numerical_cols if col not in exclude_cols]

    df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].mean())
    print(f"Imputed missing values in the following columns: {cols_to_impute}")
    return df


def drop_columns_with_missing_values(df, missing_threshold=0.5):
    threshold = len(df) * missing_threshold
    df_cleaned = df.dropna(thresh=threshold, axis=1)
    print(f"Columns dropped: {df.shape[1] - df_cleaned.shape[1]}")
    return df_cleaned


def drop_columns(df):
    df_cleaned = df.dropna(axis=0)
    return df_cleaned


def clean_vehicle_intro_date(df, column_name='VehicleIntroDate'):
    # Convert to datetime format, handling multiple formats and coercing errors to NaT
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
    
    # Format the dates to 'YYYY-MM-DD'
    df[column_name] = df[column_name].dt.strftime('%Y-%m-%d')
    
    return df

