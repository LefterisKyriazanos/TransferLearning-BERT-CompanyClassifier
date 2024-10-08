import os

# Define the directory structure
directories = [
    'data',
    'data/model_evaluation',
    'data/modelling_data',
    'data/plots',
    'data/models',
    'logs',
    'notebooks',
    'outputs',
    'src',
    'src/scripts',
    'src/scripts/db_processing',
    'src/scripts/db_setup',
    'src/scripts/eda',
    'src/scripts/helpers',
    'src/scripts/modelling',
]

# Create each directory if it doesn't exist
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

print("All directories are created or already exist.")
