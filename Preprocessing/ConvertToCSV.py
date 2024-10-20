import json;
import pandas;
from sklearn.utils import resample;

# Open the file and read line by line
with open('..\\Datasets\\NewsClassification.json', 'r') as file:
    headlines = []
    categories = []

    # Iterate over each line, assuming each line contains a JSON object
    for line in file:
        try:
            # Parse the line as a JSON object
            item = json.loads(line)
            
            # Extract the headline and category
            headlines.append(item['headline'])
            categories.append(item['category'])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

# Create a DataFrame
df = pandas.DataFrame({
    'Headline': headlines,
    'Category': categories
})

# Save to a CSV file
df.to_csv('..\\Datasets\\NewsClassification.csv', index=False)
