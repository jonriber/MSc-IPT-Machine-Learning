import pandas as pd
import random

# Create a list of drugs
drugs = ['drugY', 'drugC', 'drugX', 'drugA', 'drugB']

# Create an empty list to store the data
data = []

# Generate random data
for _ in range(14000):
    age = random.randint(20, 80)
    sex = random.choice(['M', 'F'])
    blood_pressure = random.choice(['LOW', 'NORMAL', 'HIGH'])
    cholesterol = random.choice(['NORMAL', 'HIGH'])
    na_to_k = round(random.uniform(5, 25), 3)
    drug = random.choice(drugs)
    data.append({'Age': age, 'Sex': sex, 'BP': blood_pressure, 'Cholesterol': cholesterol, 'Na_to_K': na_to_k, 'Drug': drug})

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(data)

# Export to CSV
df.to_csv('random_data.csv', index=False)