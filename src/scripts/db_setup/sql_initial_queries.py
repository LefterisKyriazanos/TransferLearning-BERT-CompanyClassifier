import sqlite3
import pandas as pd
import logging
from typing import Dict
import os

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/sql_queries.log')
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

def execute_query(query: str, output_file: str, conn: sqlite3.Connection):
    """
    Execute a SQL query and save the result to a CSV file.

    Args:
        query (str): The SQL query to execute.
        output_file (str): The path to the output CSV file.
        conn (sqlite3.Connection): The SQLite connection object.
    """
    try:
        result = pd.read_sql_query(query, conn)
        result.to_csv(output_file, index=False)
        logging.info(f"Query results saved to {output_file}")
    except Exception as e:
        logging.error(f"An error occurred while executing query: {e}")

def execute_queries(queries: Dict[str, Dict[str, str]], db_file_path):
    conn = sqlite3.connect(db_file_path)

    try:
        for task_name, task_info in queries.items():
            logging.info(f"Executing {task_name}")
            execute_query(task_info['query'], task_info['output_file'], conn)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    
    finally:
        # Ensure the database connection is closed
        conn.close()
        logging.info("Database connection closed")

if __name__ == "__main__":
    
    # set query dict for Task 1
    # add/remove key-value pairs to execute a different set of queries
    queries = {
        "Task 1: Top 10 Industries": {
            "query": """
                SELECT industry, AVG("current employee estimate") AS avg_employees
                FROM CompanyDataset
                WHERE "year founded" > 2000 
                AND "current employee estimate" > 10
                AND industry IS NOT NULL
                AND is_duplicate = 0
                GROUP BY industry
                ORDER BY avg_employees DESC
                LIMIT 10;
            """,
            "output_file": f"{os.path.join(script_dir, '../../../outputs/top_10_industries.csv')}"
        },
        "Task 2: Technology Companies with Ineffective Homepage Text": {
            "query": """
                SELECT a.CompanyName, a.industry, a."current employee estimate", b.homepage_text
                FROM CompanyDataset AS a
                RIGHT JOIN CompanyClassification AS b
                ON (a.CompanyName = b.CompanyName
                AND a.Website = b.Website)
                WHERE a.industry LIKE '%Technology%'
                AND (b.homepage_text IS NULL OR TRIM(b.homepage_text) = '')
                AND a."current employee estimate" < 100
                AND a.is_duplicate = 0
                AND b.is_duplicate = 0;
            """,
            "output_file": f"{os.path.join(script_dir, '../../../outputs/tech_companies_ineffective_homepage.csv')}"
        },
        "Task 3: Rank Companies Within Each Country": {
            "query": """
                WITH FilteredCompanies AS (
                    SELECT CompanyName, country, "total employee estimate"
                    FROM CompanyDataset
                    WHERE "total employee estimate" IS NOT NULL
                    AND country IS NOT NULL
                    AND is_duplicate = 0
                ),
                RankedCompanies AS (
                    SELECT CompanyName, country, "total employee estimate",
                        RANK() OVER (PARTITION BY country ORDER BY "total employee estimate" DESC) AS rank
                    FROM FilteredCompanies
                )
                SELECT rank, CompanyName, country, "total employee estimate"
                FROM RankedCompanies
                WHERE rank <= 5;
            """,
            "output_file": f"{os.path.join(script_dir, '../../../outputs/top_5_companies_by_country.csv')}"
        }
    }
    
    execute_queries(queries, db_file_path)

