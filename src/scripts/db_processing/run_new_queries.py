import sqlite3
import logging
import os

# Get the directory where the current script is located
script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/company_data.log')
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

class CompanyDataDB:
    def __init__(self, db_path):
        self.db_path = db_path

    def _connect(self):
        """Private method to connect to the database."""
        return sqlite3.connect(self.db_path)

    def insert_records(self, records):
        """Insert multiple records into the database."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info("Inserting records into the database.")

        query = '''
        INSERT INTO MergedCompanyData (
            CompanyName, Website, industry, "current employee estimate",
            "total employee estimate", "year founded", "size range",
            country, locality, "linkedin url", Category, homepage_text,
            h1, h2, h3, nav_link_text, meta_keywords, meta_description
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''

        try:
            cursor.executemany(query, records)
            conn.commit()
            logging.info(f"Inserted {cursor.rowcount} records successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error inserting records: {e}")
            raise
        finally:
            conn.close()

    def delete_records(self, condition):
        """Delete records from the database based on the provided condition."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Deleting records from the database with condition: {condition}")

        query = f'''
        DELETE FROM MergedCompanyData WHERE {condition};
        '''

        try:
            cursor.execute(query)
            conn.commit()
            logging.info(f"Deleted {cursor.rowcount} records successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error deleting records: {e}")
            raise
        finally:
            conn.close()

    def get_companies_by_industry(self, industry_name):
        """Retrieve companies based on industry."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Retrieving companies in industry: {industry_name}")

        query = '''
        SELECT * FROM MergedCompanyData WHERE industry = ?;
        '''
        try:
            cursor.execute(query, (industry_name,))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving companies by industry: {e}")
            raise
        finally:
            conn.close()

    def get_companies_by_country_and_size(self, country, size_range):
        """Retrieve companies based on country and size range."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Retrieving companies in country: {country} with size range: {size_range}")

        query = '''
        SELECT CompanyName, Website, "current employee estimate", "total employee estimate", "size range", country 
        FROM MergedCompanyData WHERE country = ? AND "size range" = ?;
        '''
        try:
            cursor.execute(query, (country, size_range))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving companies by country and size: {e}")
            raise
        finally:
            conn.close()

    def search_companies_by_name(self, search_term):
        """Search companies by partial CompanyName."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Searching for companies with name containing: {search_term}")

        query = '''
        SELECT * FROM MergedCompanyData WHERE CompanyName LIKE ?;
        '''
        try:
            cursor.execute(query, ('%' + search_term + '%',))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error searching companies by name: {e}")
            raise
        finally:
            conn.close()

    def get_companies_founded_after(self, year):
        """Retrieve companies founded after a specific year."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Retrieving companies founded after the year: {year}")

        query = '''
        SELECT CompanyName, Website, "year founded" FROM MergedCompanyData WHERE "year founded" > ?;
        '''
        try:
            cursor.execute(query, (year,))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving companies founded after a specific year: {e}")
            raise
        finally:
            conn.close()

    def paginate_companies(self, limit, offset):
        """Retrieve companies with pagination."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Retrieving companies with pagination: limit {limit}, offset {offset}")

        query = '''
        SELECT * FROM MergedCompanyData ORDER BY id LIMIT ? OFFSET ?;
        '''
        try:
            cursor.execute(query, (limit, offset))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error paginating companies: {e}")
            raise
        finally:
            conn.close()

    def count_companies_by_industry(self):
        """Count companies per industry."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info("Counting companies by industry.")

        query = '''
        SELECT industry, COUNT(*) AS company_count FROM MergedCompanyData GROUP BY industry ORDER BY company_count DESC;
        '''
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            logging.info(f"Retrieved counts for {len(results)} industries successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error counting companies by industry: {e}")
            raise
        finally:
            conn.close()

    def get_companies_by_metadata_keyword(self, keyword):
        """Retrieve companies with a specific keyword in metadata."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info(f"Retrieving companies with metadata keyword: {keyword}")

        query = '''
        SELECT CompanyName, Website, meta_keywords 
        FROM MergedCompanyData WHERE meta_keywords LIKE ?;
        '''
        try:
            cursor.execute(query, ('%' + keyword + '%',))
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving companies by metadata keyword: {e}")
            raise
        finally:
            conn.close()

    def create_tech_companies_usa_view(self):
        """Create a view for Tech Companies in the USA."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info("Creating view for Tech Companies in the USA.")

        query = '''
        CREATE VIEW IF NOT EXISTS TechCompaniesUSA AS
        SELECT CompanyName, Website, industry, country, "year founded" 
        FROM MergedCompanyData 
        WHERE industry = 'Technology' AND country = 'USA';
        '''
        try:
            cursor.execute(query)
            conn.commit()
            logging.info("View created successfully.")
        except sqlite3.Error as e:
            logging.error(f"Error creating view: {e}")
            raise
        finally:
            conn.close()

    def get_tech_companies_usa(self):
        """Retrieve companies from the TechCompaniesUSA view."""
        conn = self._connect()
        cursor = conn.cursor()
        logging.info("Retrieving companies from the TechCompaniesUSA view.")

        query = '''
        SELECT * FROM TechCompaniesUSA;
        '''
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            logging.info(f"Retrieved {len(results)} companies successfully.")
            return results
        except sqlite3.Error as e:
            logging.error(f"Error retrieving companies from TechCompaniesUSA view: {e}")
            raise
        finally:
            conn.close()

def main():
    db = CompanyDataDB(db_file_path)
    
    # Example 1: Insert records (add your own records here)
    records = [
        ('Company A', 'www.companya.com', 'Technology', 100, 150, 2010, '100-150', 'USA', 'New York', 'www.linkedin.com/companya', 'Software', 'Homepage text A', 'H1 A', 'H2 A', 'H3 A', 'Nav A', 'Keywords A', 'Description A'),
        ('Company B', 'www.companyb.com', 'Healthcare', 200, 250, 2005, '200-250', 'USA', 'San Francisco', 'www.linkedin.com/companyb', 'Medical', 'Homepage text B', 'H1 B', 'H2 B', 'H3 B', 'Nav B', 'Keywords B', 'Description B'),
    ]
    # db.insert_records(records)
    
    # Example 2: Retrieve companies by industry
    tech_companies = db.get_companies_by_industry('Technology')
    print("Technology Companies:", tech_companies)
    
    # Example 3: Delete records by condition
    # db.delete_records("industry = 'Healthcare'")
    
    # Example 4: Search companies by name
    search_results = db.search_companies_by_name('Tech')
    print("Companies with 'Tech' in their name:", search_results)
    
    # Continue with other operations...
    db.get_companies_by_metadata_keyword('software')
    print("Companies with software metadata_keyword", search_results)
if __name__ == "__main__":
    main()
