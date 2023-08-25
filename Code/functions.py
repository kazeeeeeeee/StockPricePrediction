import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error

# Use in "2. Data Cleaning and Basic EDA.ipynb"

def plot_null_values_features_of_df(df):    
    # Create a figure and axis object with two subplots (two rows)
    fig, axes = plt.subplots(nrows=2, figsize=(16, 18))

    # Get the first and second halves of the columns (parameters)
    half_columns_1st = df.columns[:len(df.columns) // 2]
    half_columns_2nd = df.columns[len(df.columns) // 2:]

    # Plot the first half of the DataFrame in the first subplot
    # Lists to store counts and percentages for each parameter
    null_value_counts_list_1st = []
    null_value_percentages_list_1st = []

    # Loop through each column (parameter) in the first half of the DataFrame
    for para in half_columns_1st:
        null_value_counts = df[para].isnull().sum()
        null_value_counts_list_1st.append(null_value_counts)

        # Calculate the percentage of null values for each parameter
        total_count = len(df)
        null_value_percentage = (null_value_counts / total_count) * 100
        null_value_percentages_list_1st.append(null_value_percentage)

        # Plot the count of null values on the primary y-axis (left)
        bars = axes[0].bar(para, null_value_counts, label=f'{para} (Null Count)')

    # Create a secondary y-axis (right) for the percentages in the first subplot
    ax2_1 = axes[0].twinx()
    line, = ax2_1.plot(half_columns_1st, null_value_percentages_list_1st, color='r', marker='o', label='Percentage of Null Values')
    ax2_1.set_ylabel('Percentage of Null Values (%)', color='r')
    ax2_1.tick_params(axis='y', labelcolor='r')

    # Annotate the line with percentage values (rounded to 1 decimal place) in the first subplot
    for x, y in zip(half_columns_1st, null_value_percentages_list_1st):
        ax2_1.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', color='r')
    
    # Set labels and title for the first subplot
    axes[0].set_xlabel('Parameter')
    axes[0].set_ylabel('Count of Null Values')
    axes[0].set_title('Count and Percentage of Null Values for Each Parameter (1st half of dataframe)')

    # Set the tick positions and labels for the x-axis with vertical orientation in the first subplot
    axes[0].set_xticks(range(len(half_columns_1st)))
    axes[0].set_xticklabels([f'{para}' for para in half_columns_1st], rotation='vertical')

    # Plot the second half of the DataFrame in the second subplot
    # Lists to store counts and percentages for each parameter
    null_value_counts_list_2nd = []
    null_value_percentages_list_2nd = []

    # Loop through each column (parameter) in the second half of the DataFrame
    for para in half_columns_2nd:
        null_value_counts = df[para].isnull().sum()
        null_value_counts_list_2nd.append(null_value_counts)

        # Calculate the percentage of null values for each parameter
        total_count = len(df)
        null_value_percentage = (null_value_counts / total_count) * 100
        null_value_percentages_list_2nd.append(null_value_percentage)

        # Plot the count of null values on the primary y-axis (left)
        bars = axes[1].bar(para, null_value_counts, label=f'{para} (Null Count)')

    # Create a secondary y-axis (right) for the percentages in the second subplot
    ax2_2 = axes[1].twinx()
    line, = ax2_2.plot(half_columns_2nd, null_value_percentages_list_2nd, color='r', marker='o', label='Percentage of Null Values')
    ax2_2.set_ylabel('Percentage of Null Values (%)', color='r')
    ax2_2.tick_params(axis='y', labelcolor='r')

    # Annotate the line with percentage values (rounded to 1 decimal place) in the second subplot
    for x, y in zip(half_columns_2nd, null_value_percentages_list_2nd):
        ax2_2.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 5), textcoords='offset points',
                       ha='center', va='bottom', color='r')
    
    # Set labels and title for the second subplot
    axes[1].set_xlabel('Parameter')
    axes[1].set_ylabel('Count of Null Values')
    axes[1].set_title('Count and Percentage of Null Values for Each Parameter (2nd half of dataframe)')

    # Set the tick positions and labels for the x-axis with vertical orientation in the second subplot
    axes[1].set_xticks(range(len(half_columns_2nd)))
    axes[1].set_xticklabels([f'{para}' for para in half_columns_2nd], rotation='vertical')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Use in "3. EDA.ipynb"

def get_min_max_date_for_stock_list (df, stock_list):
    data = {'symbol': [],
            'min_date': [],
            'max_date': []}
    for symbol in stock_list:
        data['symbol'].append(symbol)
        data['min_date'].append(df.loc[df['symbol']==symbol].index.min())
        data['max_date'].append(df.loc[df['symbol']==symbol].index.max())
    data_df = pd.DataFrame(data)
    return data_df
        
def plot_values_by_symbol(df, symbol_list, param):
    # Create a figure and axis object with subplots based on the number of symbols
    num_symbols = len(symbol_list)
    fig, axes = plt.subplots(nrows=num_symbols, figsize=(16, 6*num_symbols))

    for i, symbol in enumerate(symbol_list):
        symbol_data = df.loc[df['symbol'] == symbol]

        if not symbol_data.empty:  # Check if there is data for the current symbol
            # Plot the 'close' values for the current symbol in the respective subplot
            axes[i].plot(symbol_data.index, symbol_data[param], linestyle='-', label='Close Value')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Close Value')
            axes[i].set_title(f'Close Values for Symbol: {symbol}')
            axes[i].legend()

            # Set the tick positions and labels for the x-axis with rotation
            axes[i].tick_params(axis='x', rotation=45)

            # Set x-axis limits based on the available data for the symbol
            min_date = symbol_data.index.min()
            max_date = symbol_data.index.max()
            axes[i].set_xlim(min_date, max_date)
        else:
            # If there is no data for the current symbol, set x-axis limits to cover the entire data range
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Close Value')
            axes[i].set_title(f'No Data for Symbol: {symbol}')
            axes[i].text(0.5, 0.5, "No data available for this symbol.", ha='center', va='center', transform=axes[i].transAxes)

    # Adjust the layout and spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_min_max_date_for_hist_data (data_df):
    # Set up the plot
    plt.figure(figsize=(10, 10))

    # Loop through each row in the DataFrame and plot a line for each symbol
    for index, row in data_df.iterrows():
        symbol = row['symbol']
        min_date = row['min_date']
        max_date = row['max_date']
    
        # Create a line plot for the current symbol from min_date to max_date
        plt.plot([min_date, max_date], [symbol, symbol], label=symbol)

    # Set the y-axis ticks to show the symbols as strings
    plt.yticks(data_df['symbol'])

    # Add labels and legend
    plt.xlabel('Date')
    plt.ylabel('Symbol')

    # Show the plot
    plt.show()

    
def plot_moving_average_by_symbol (df, symbol, parameter):
    plt.figure(figsize=(12, 8))
    plt.plot(df.loc[df['symbol']==symbol][parameter], label=parameter)  # Line plot for 'close'
    plt.plot(df.loc[df['symbol']==symbol][parameter].rolling(2).mean(), label='2MA')  # Line plot for 2-mth moving average
    plt.plot(df.loc[df['symbol']==symbol][parameter].rolling(3).mean(), label='3MA')  # Line plot for 3-mth moving average
    plt.plot(df.loc[df['symbol']==symbol][parameter].rolling(6).mean(), label='6MA')  # Line plot for 6-mth moving average
    plt.plot(df.loc[df['symbol']==symbol][parameter].rolling(9).mean(), label='9MA')  # Line plot for 9-mth moving average
    plt.plot(df.loc[df['symbol']==symbol][parameter].rolling(12).mean(), label='12MA')  # Line plot for 12-mth moving average
    plt.title(f'Moving Average of: {parameter} for {symbol}')  # Plot title
    plt.legend()
    plt.show()
    
def plot_value_difference_between_each_column_against_previous_by_symbol (df, symbol, parameter):
    # defining rows and canvas size for plots
    fig, axes = plt.subplots(nrows=3, figsize=(15,12))

    # plot 1
    axes[0].plot(df.loc[df['symbol']==symbol][parameter], label='Close')
    axes[0].legend()

    # plot 2
    axes[1].plot(df.loc[df['symbol']==symbol][parameter].diff(), label='Diff')
    axes[1].legend()

    # plot 3
    axes[2].plot(df.loc[df['symbol']==symbol][parameter].pct_change(), label='Pct Diff')
    axes[2].legend()
    
    
    
# Use in 4. Preprocessing and Modelling

def create_time_lagged_columns_df (df, num_lagged_columns, parameter):

    # Specify the number of time-lagged columns you want to create i.e. num_lagged_columns, e.g. 3
    
    # Create time-lagged columns
    for lag in range(1, num_lagged_columns + 1):
        df[f'{parameter}_t_minus_{lag}'] = df.groupby('symbol')[parameter].shift(lag)
        
        
def train_test_split_for_model (df, symbol, target_y, test_size_value):
    X = df.loc[df['symbol']==symbol].drop(columns = [target_y,'symbol','currency','year','month','day','date'])
    y = df.loc[df['symbol']==symbol][target_y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size_value, shuffle=False)
    return X, X_train, X_test, y_train, y_test


def standardise_features (df, symbol, X_train, X_test):
    
    # Create an instance of the StandardScaler
    ss = StandardScaler()

    # Fit the scaler to the train data
    ss.fit(X_train)

    # Transform the train and test data using the fitted scaler
    X_train_scaled = ss.transform(X_train)
    X_test_scaled = ss.transform(X_test)
    
    return X_train_scaled, X_test_scaled


def linear_model (X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    # initiate Linear Regression model
    lr = LinearRegression()

    # fit the model
    lr.fit(X_train_scaled,y_train)
    
    # Make predictions
    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    
    rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Use R-square value to evaluate performance
    lr_score_train = lr.score(X_train_scaled,y_train)
    lr_score_test = lr.score(X_test_scaled,y_test)
    
    # Create dataframe of the predicted result
    y_pred_test_df = pd.DataFrame({'predicted_close':y_pred_test,'symbol':'TSLA'},index =X_test.index)
    y_pred_test_df.head()
    
    lr_intercept = lr.intercept_
    lr_coeff = lr.coef_
    
    return y_pred_train, y_pred_test, rmse_train_lr, rmse_test_lr, lr_score_train, lr_score_test, y_pred_test_df, lr_intercept, lr_coeff


def correlation_matrix_df (df):
    # Select only the numeric columns for calculating correlations
    numeric_columns = df.select_dtypes(include=['float64', 'int64'])

    # Calculate the correlation matrix
    corr_matrix = numeric_columns.corr()

    # Drop the diagonal elements (replace them with NaN)
    corr_matrix.values[np.triu_indices(corr_matrix.shape[0])] = np.nan

    # Create a mask for upper triangular part
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Apply the mask to the correlation matrix
    masked_corr_matrix = corr_matrix.mask(mask)

    # Create a DataFrame to store correlation pairs and their values
    correlation_pairs = []
    correlation_values = []

    # Iterate through the masked correlation matrix
    for row in masked_corr_matrix.index:
        for col in masked_corr_matrix.columns:
            correlation_value = masked_corr_matrix.loc[row, col]
            if not pd.isnull(correlation_value):
                correlation_pairs.append(f'{row} - {col}')
                correlation_values.append(round(correlation_value, 5))  # Round to 5 decimal places

    # Create a DataFrame
    correlation_df = pd.DataFrame({'Correlation_Pair': correlation_pairs, 'Correlation_Value': correlation_values})

    # Sort the DataFrame by correlation values in descending order
    correlation_df = correlation_df.sort_values(by='Correlation_Value', ascending=False)

    # Return the sorted DataFrame
    return correlation_df
