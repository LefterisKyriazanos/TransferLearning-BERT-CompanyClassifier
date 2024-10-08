import pandas as pd
import sqlite3
import logging
import os

from ..helpers import text_helper

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/create_merged_table.log')
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

def load_merged_data(db_file_path):
    """Merge the datasets in SQL and load the merged data into a pandas DataFrame."""
    logging.info("Starting to load and merge data from SQL.")
    conn = sqlite3.connect(db_file_path)
    
    # SQL query to merge datasets, excluding the 'Unnamed: 0' column
    query = """
    SELECT 
        -- common between tables
        a.CompanyName, 
        a.Website, 
        
        -- CompanyDataset
        a.industry, 
        a."current employee estimate", 
        a."total employee estimate",
        a."year founded", 
        a."size range",
        a.country,
        a.locality,
        a."linkedin url",
        -- a.is_duplicate AS CompanyDataset_is_duplicate,

        -- CompanyClassification
        b.Category,
        b.homepage_text, 
        b.h1, 
        b.h2,
        b.h3,
        b.nav_link_text,
        b.meta_keywords,
        b.meta_description
        -- b.is_duplicate AS CompanyClassification_is_duplicate
    FROM 
        CompanyDataset AS a
    RIGHT JOIN 
        CompanyClassification AS b
    ON 
        (a.CompanyName = b.CompanyName AND a.Website = b.Website)
    WHERE 
        a.is_duplicate = 0 
        AND b.is_duplicate = 0
        -- exclude rows that correspond to websites that point to more than one company or category
        AND a.Website NOT IN (
            SELECT Website 
            FROM CompanyClassification
            GROUP BY Website
            HAVING COUNT(DISTINCT CompanyName) > 1 
                OR COUNT(DISTINCT Category) > 1
        );

    """
    # Load the result of the SQL query into a pandas DataFrame
    merged_data = pd.read_sql_query(query, conn)
    conn.close()
    logging.info("Data merged and loaded successfully.")
    
    # Assert uniqueness on common keys/ company identifiers
    assert merged_data[merged_data.duplicated(subset=['CompanyName', 'Website', 'Category'], keep=False)].shape[0]  == 0
    logging.info("Data uniqueness ('CompanyName' x 'Website' x 'Category') checks passed.")
    
    return merged_data

def create_table_to_db(merged_data, db_file_path):
    """Save the cleaned and prepared merged data into a new table in the SQLite database."""
    logging.info("Starting to save data to the database.")
    conn = sqlite3.connect(db_file_path)
    
    # Drop the MergedCompanyData table if it exists to recreate it
    conn.execute('DROP TABLE IF EXISTS MergedCompanyData;')
    
    # Create the new table with an auto-incrementing primary key
    conn.execute('''
    CREATE TABLE MergedCompanyData (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        CompanyName TEXT,
        Website TEXT,
        industry TEXT,
        "current employee estimate" INTEGER,
        "total employee estimate" INTEGER,
        "year founded" INTEGER,
        "size range" TEXT,
        country TEXT,
        locality TEXT,
        "linkedin url" TEXT,
        Category TEXT,
        homepage_text TEXT,
        h1 TEXT,
        h2 TEXT,
        h3 TEXT,
        nav_link_text TEXT,
        meta_keywords TEXT,
        meta_description TEXT,
        meta_description_is_english REAL,  -- [0,1]
        UNIQUE(CompanyName, Website, Category)  -- Unique constraint for the combination
    );
    ''')
    
    # Insert data into the newly created table
    merged_data.to_sql('MergedCompanyData', conn, if_exists='append', index=False)
    logging.info("Data inserted into the database successfully.")
    
    # Create indexes on important columns
    indexes = [
        'CompanyName', 
        'Website', 
        'industry', 
        'Category', 
        'country', 
        '"year founded"', 
        '"current employee estimate"', 
        '"total employee estimate"',
        '"linkedin url"'
    ]
    
    for index in indexes:
        conn.execute(f'''
        CREATE INDEX IF NOT EXISTS idx_{index.replace(" ", "_").replace('"', '')} 
        ON MergedCompanyData ({index});
        ''')
    
    conn.commit()
    logging.info("Indexes created successfully.")
    conn.close()

def process_text_columns(merged_data, webpage_columns, text_columns):
    """
    Process the specified columns in the merged data:
    - Cleans and processes the webpage-related columns.
    - Normalizes the other text columns.

    Args:
    merged_data (pd.DataFrame): The DataFrame containing the merged data.
    webpage_columns (list): List of webpage-related columns to be cleaned.
    text_columns (list): List of text columns to be normalized.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Preprocess text columns (webpage columns)
    for column in webpage_columns:
        merged_data[column] = merged_data[column].apply(lambda x: text_helper.clean_text(x) if x is not None else '')
        merged_data[column] = merged_data[column].replace('', pd.NA)
    logging.info(f"Webpage columns {webpage_columns} preprocessed successfully.")
        
    # Normalize the remaining text columns
    for column in text_columns:
        merged_data[column] = merged_data[column].apply(lambda x: text_helper.normalize_text(x) if x is not None else '')
        merged_data[column] = merged_data[column].replace('', pd.NA)
    logging.info(f"Text columns {text_columns} normalized successfully.")
    
    return merged_data
   
def main():
    logging.info("Process started.")
    webpage_columns = ['homepage_text', 'h1', 'h2', 'h3', 'nav_link_text', 'meta_keywords', 'meta_description']
    text_columns = ['industry', 'country', 'locality', 'linkedin url', 'Category']
    
    # Load, merge, and clean data
    merged_data = load_merged_data(db_file_path)
    
    logging.info("Processing Columns")
    # Set 'year founded' type to int
    merged_data['year founded'] = merged_data['year founded'].fillna(0).astype(int)
    
    # Process columns
    merged_data = process_text_columns(merged_data, webpage_columns, text_columns)
    
    # flag any non - english text
    merged_data['meta_description_is_english'] = merged_data['meta_description'].apply(lambda x: text_helper.is_english(x) if pd.notna(x) else pd.NA)
        
    # Save the cleaned data back to the database
    create_table_to_db(merged_data, db_file_path)
    logging.info("Process completed successfully.")

if __name__ == "__main__":
    main()