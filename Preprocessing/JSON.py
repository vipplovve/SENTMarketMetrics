import json;

RawData = [];

with open('..\\Datasets\\FinancialPhraseBankCombined.txt', 'r') as file:
    
    lines = file.readlines();

    for line in lines:

        RawData.append(line);
            
print("\nData Loaded Successfully");

Json = [];

for sentence in RawData:
    
    text, label = sentence.rsplit('@', 1);
    
    Json.append({"Text": text.strip(), "Verdict": label.strip().lower()});

Output = json.dumps(Json, indent=4);

with open('..\\Datasets\\FinancialPhraseBankCombined.json', 'w') as json_file:
    json_file.write(Output);
    
print("\nData Saved Successfully");