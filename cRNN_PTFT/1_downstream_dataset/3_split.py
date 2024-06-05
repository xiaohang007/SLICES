import pandas as pd
from sklearn.model_selection import train_test_split
import json

def split_data(json_file, train_file, test_file, test_size):
    try:
        # Load JSON data
        with open(json_file, 'r') as file:
            df = pd.read_csv(file,header=None)

        # Split the data into training and testing sets
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=10)
        train_df.to_csv(train_file, header=None, index=None)
        test_df.to_csv(test_file, header=None, index=None)

        print(f"Data split into training and testing sets and saved to {train_file} and {test_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Paths to your files
json_file = 'downstream.sli'  # Update with the path to your JSON file
train_file = 'train_downstream.sli'  # Update with the path to save your training data
test_file = 'test_downstream.sli'  # Update with the path to save your testing data
# Run the function
split_data(json_file, train_file, test_file, 0.1)
