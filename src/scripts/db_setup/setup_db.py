import sqlite3
import pandas as pd
import logging
import os

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/setup_db.log')
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

def standardize_columns(conn):
    cursor = conn.cursor()

    # Standardize CompanyName and Website in CompanyDataset
    logging.info("Standardizing columns in CompanyDataset")
    cursor.execute("""
        UPDATE CompanyDataset
        SET CompanyName = LOWER(TRIM(CompanyName)),
            Website = LOWER(TRIM(Website))
    """)

    # Standardize CompanyName and Website in CompanyClassification
    logging.info("Standardizing columns in CompanyClassification")
    cursor.execute("""
        UPDATE CompanyClassification
        SET CompanyName = LOWER(TRIM(CompanyName)),
            Website = LOWER(TRIM(Website))
    """)


def add_is_duplicate_column(conn, table_name):
    """
    Add a new column 'is_duplicate' to the table if it does not exist.
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'is_duplicate' not in columns:
        logging.info(f"Adding 'is_duplicate' column to {table_name}")
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN is_duplicate INTEGER DEFAULT 0;")
        
def update_duplicates(conn, table_name, df):
    """
    Flag duplicates in the DataFrame based on 'CompanyName' and 'Website' columns,
    and update only the rows flagged as duplicates in the database.
    The record with the fewest missing values is kept, and the rest are flagged.
    """
    # Calculate the number of missing values for each row
    df['missing_values_count'] = df.isnull().sum(axis=1)

    if table_name == 'CompanyDataset':
        # Sort based on 'CompanyName', 'Website', and 'missing_values_count' to prioritize records with fewer missing values
        # If these criteria match between rows then prioritize rows with higher 'total employee estimate' and most recent 'year founded'
        df = df.sort_values(by=['CompanyName', 'Website', 'missing_values_count', 'total employee estimate', 'year founded'], ascending=[True, True, True, False, False])
    elif table_name == 'CompanyClassification':
        # If the first 3 criteria match then prioritize rows non-null 'meta_description' or 'homepage_text'
        df = df.sort_values(by=['CompanyName', 'Website', 'missing_values_count', 'meta_description', 'homepage_text'], ascending=[True, True, True, True, True])
        
    # Flag duplicates - keep the first occurrence (the one with the fewest missing values)
    df['is_duplicate'] = df.duplicated(subset=['CompanyName', 'Website'], keep='first').astype(int)

    # Filter only the rows where is_duplicate is 1
    duplicates_df = df[df['is_duplicate'] == 1]

    # Update the is_duplicate column only for the flagged duplicates in the database
    cursor = conn.cursor()
    for index, row in duplicates_df.iterrows():
        cursor.execute(f"UPDATE {table_name} SET is_duplicate = 1 WHERE ROWID = ?", (row['rowid'],))
    
    logging.info(f"Flagged and updated duplicates in {table_name}")

def flag_duplicates(conn):
    """
    Process both tables by adding the 'is_duplicate' column and flagging duplicates.
    """
    tables = ['CompanyDataset', 'CompanyClassification']

    for table in tables:
        logging.info(f"Processing table: {table}")
        
        # Add the is_duplicate column if it doesn't exist
        add_is_duplicate_column(conn, table)

        # Load the table into a DataFrame with ROWID
        query = f"SELECT *, ROWID FROM {table};"
        df = pd.read_sql_query(query, conn)

        # Flag duplicates and update the table
        update_duplicates(conn, table, df)
        
def create_indexes(conn):
    cursor = conn.cursor()

    logging.info("Creating indexes for CompanyDataset")
    # Indexes for CompanyDataset
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_year_founded ON CompanyDataset('year founded');")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_current_employee_estimate ON CompanyDataset('current employee estimate');")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_industry ON CompanyDataset(industry);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_company_name ON CompanyDataset(CompanyName);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_website ON CompanyDataset(Website);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country_total_employee ON CompanyDataset(country, 'total employee estimate');")

    logging.info("Creating indexes for CompanyClassification")
    # Indexes for CompanyClassification
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_company_name ON CompanyClassification(CompanyName);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_class_website ON CompanyClassification(Website);")

if __name__ == "__main__":
    conn = sqlite3.connect(db_file_path)
    
    try:
        logging.info("Starting the setup_db process")

        # Standardize columns on both tables (CompanyName, Website)
        standardize_columns(conn)
        
        # Process tables to add is_duplicate column and flag duplicates
        flag_duplicates(conn)
        
        # Create indexes on frequently filtered columns
        create_indexes(conn)

        # Commit all changes as a single transaction
        conn.commit()
        logging.info("All changes committed successfully")

    except Exception as e:
        # Rollback any changes if something goes wrong
        conn.rollback()
        logging.error(f"An error occurred: {e}")
    
    finally:
        conn.close()
        logging.info("Database connection closed")
