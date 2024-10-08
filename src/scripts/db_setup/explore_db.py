import sqlite3
import pandas as pd
import logging
import os


# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/explore_db.log')
db_file_path = os.path.join(script_dir, '../../../data/combined_data.db')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
        logging.StreamHandler()
    ]
)

def get_table_size(cursor, table_name):
    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    return cursor.fetchone()[0]

def get_missing_values_per_column(cursor, table_name, column_name):
    cursor.execute(f'SELECT COUNT(*) - COUNT("{column_name}") AS missing_values FROM "{table_name}"')
    return cursor.fetchone()[0]

def get_rows_with_any_missing_values(cursor, table_name, columns):
    conditions = " OR ".join([f'"{col}" IS NULL' for col in columns])
    cursor.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE {conditions}')
    return cursor.fetchone()[0]

def fetch_example_value(cursor, table_name, column_name):
    cursor.execute(f'SELECT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL LIMIT 1')
    result = cursor.fetchone()
    return result[0] if result and len(result) > 0 else 'All Values are NaN'

def check_foreign_keys(cursor, table_name):
    cursor.execute(f'PRAGMA foreign_key_list("{table_name}")')
    return cursor.fetchall()

def explore_db(db_file_path: str):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    
    # List all tables in the database
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    logging.info("Tables in the database:")
    logging.info(tables['name'].tolist())
    
    # Combine schema description and data overview in a single loop
    for table_name in tables['name']:
        logging.info(f"{'='*20}")
        logging.info(f"Exploring table: {table_name}")
        logging.info(f"{'='*20}")
        # Get table size
        total_rows = get_table_size(cursor, table_name)
        
        schema = cursor.execute(f"PRAGMA table_info({table_name});").fetchall()
        logging.info(f"Schema for {table_name}:")
        logging.info(f"Total Rows: {total_rows}")
        logging.info(f"{'='*20}")
        for column in schema:
            logging.info(f"Column Name: {column[1]}")
            logging.info(f"Data Type: {column[2]}")
            # Check if the column is a primary key
            is_primary_key = "YES" if column[5] > 0 else "NO"
            logging.info(f"Primary Key: {is_primary_key}")
            col_name = column[1]
            missing_values = get_missing_values_per_column(cursor, table_name, col_name)
            logging.info(f"{missing_values} missing values")
            example_value = fetch_example_value(cursor, table_name, col_name)
            logging.info(f"Example Value -> {example_value}\n")
            
          # Check for foreign keys
        foreign_keys = check_foreign_keys(cursor, table_name)
        if foreign_keys:
            logging.info(f"Foreign Keys in {table_name}:")
            for fk in foreign_keys:
                logging.info(fk)
        else:
            logging.info(f"No Foreign Keys in {table_name}\n")

        # Get rows with any missing values
        columns = [column[1] for column in schema]
        rows_with_na = get_rows_with_any_missing_values(cursor, table_name, columns)
        logging.info(f"Total Rows with Any Missing Values: {rows_with_na}\n")

    # Close the connection
    conn.close()

if __name__ == "__main__":
    explore_db(db_file_path)
