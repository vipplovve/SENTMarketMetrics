lines = int(input("Enter How Many Headlines You'd Be Entering: "));

with open("Prompts.txt", "w") as file:
    
    for i in range(lines):
        
        file.write(input(f"Enter Headline {i + 1}: ") + "\n");
        
print("Headlines Entered Successfully!");

print("Data Has Been Saved To 'Prompts.txt' File!");