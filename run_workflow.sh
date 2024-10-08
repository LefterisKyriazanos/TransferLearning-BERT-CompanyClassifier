#!/bin/bash

# Stop the script if any command fails
set -e

echo "Starting the workflow..."

# Step 1: Install the required packages
echo "Installing dependencies..."
pip install -r requirements.txt

# Create Dirs (if they don't exist)
echo "Running database exploration script..."
python make_dirs.py

# Step 2: Explore the database
echo "Creating Directories..."
python -m src.scripts.db_setup.explore_db

# Step 3: Set up the database
echo "Setting up the database..."
python -m src.scripts.db_setup.setup_db

# Step 4: Run the initial SQL queries
echo "Running initial SQL queries..."
python -m src.scripts.db_setup.sql_initial_queries

# Step 5: Merge tables in the database
echo "Merging tables in the database..."
python -m src.scripts.db_processing.merge_tables

# Step 6: Run new queries on the database
echo "Running new queries on the database..."
python -m src.scripts.db_processing.run_new_queries

# Step 7: Perform EDA and data cleaning
echo "Performing EDA and data cleaning..."
python -m src.scripts.eda.eda_and_data_cleaning

##
# Comment out if your environment has sufficient resources to train the model
## 

# Step 8: Train the machine learning model
# echo "Training the machine learning model..."
# python -m src.scripts.modelling.ml_model_training

echo "Workflow completed successfully!"
