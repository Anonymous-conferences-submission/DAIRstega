import json

# Load the user's JSON file
file_path = './data_json/steganalysis_data.json'


# Function to count tokens in a JSON object
def count_tokens(obj):
    if isinstance(obj, dict):
        return sum([count_tokens(key) + count_tokens(value) for key, value in obj.items()])
    elif isinstance(obj, list):
        return sum([count_tokens(item) for item in obj])
    else:
        # Assuming each key and value is a single token
        return 1

# Read and parse the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Count the tokens in the JSON data
token_count = count_tokens(data)
print(token_count)
