import json
import pandas as pd

# Assuming your JSON file is named 'your_file.json'
with open('./data_json/3 theme.json', 'r') as file:
    json_data = json.load(file)

# Converting JSON data to a pandas DataFrame
df = pd.DataFrame(json_data)

# Saving the DataFrame to a CSV file
df.to_csv('./data_json/theme.csv', index=False)
