import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging
from ..helpers import plot_helper

script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/eda_and_data_cleaning.log')
db_file_path = os.path.join(script_dir, '../../../data/combined_data.db')
file_outputs = os.path.join(script_dir, '../../../outputs')
data_modelling_path = os.path.join(script_dir, '../../../data/modelling_data')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
        logging.StreamHandler()
    ]
)


def tfidf_top_terms(data, category_col, text_col, top_n=10, output_file=f"{file_outputs}/tfidf_top_terms.csv"):
    """
    Calculates and saves the top N terms by TF-IDF for each category to a CSV file.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    category_col (str): The name of the column containing the categories.
    text_col (str): The name of the column containing the text.
    top_n (int): The number of top terms to display for each category.
    output_file (str): The name of the output CSV file.

    Returns:
    None
    """
    categories = data[category_col].unique()
    all_top_terms = []

    for category in categories:
        # Filter the data for the current category
        category_data = data[data[category_col] == category]
        
        # Combine all text from the text column into one string
        combined_text = category_data[text_col].dropna().tolist()
        
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(combined_text)
        
        # Sum TF-IDF values across all documents (rows)
        sum_tfidf = tfidf_matrix.sum(axis=0)
        term_scores = pd.DataFrame(sum_tfidf.T, index=vectorizer.get_feature_names_out(), columns=["tfidf"])
        
        # Get the top N terms
        top_terms = term_scores.sort_values(by="tfidf", ascending=False).head(top_n)
        top_terms.reset_index(inplace=True)
        top_terms.columns = ["term", "tfidf"]
        top_terms["category"] = category
        
        # Append to the list of all top terms
        all_top_terms.append(top_terms)
    
    # Concatenate all top terms into a single DataFrame
    final_df = pd.concat(all_top_terms, ignore_index=True)
    
    # Save the DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    
def flag_outliers(df, columns):
    """
    Flags outliers in specified columns using the IQR method.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to check for outliers.
    
    Returns:
    pd.DataFrame: A DataFrame with a new column for each input column, 
                  indicating whether the value is an outlier (True/False).
    """
    outlier_flags = pd.DataFrame(index=df.index)  # To store the outlier flags
    
    for col in columns:
        # Filter out zero-length values
        non_zero_data = df[df[col] > 0][col]
        
        # Calculate Q1, Q3, and IQR
        Q1 = non_zero_data.quantile(0.25)
        Q3 = non_zero_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Flag outliers
        df[f'{col}_outlier'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    
    return df
   
def main():
    # connect to db 
    logging.info("Connecting to the database.")
    conn = sqlite3.connect(db_file_path) 
    merged_company_data = pd.read_sql_query('''Select * from MergedCompanyData
                                        where meta_description is not NULL;''', conn)
    conn.close()
    logging.info(f"Successfully fetched data from the database. Shape of the dataset: {merged_company_data.shape}")

    logging.info('Starting meta_description deep dive.')

    # calculate len 
    logging.info("Calculating the length of meta descriptions.")
    merged_company_data['meta_description_length'] = merged_company_data['meta_description'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    logging.info(f"Meta description length statistics:\n{merged_company_data['meta_description_length'].describe()}")

    # Calculate the 10th and 90th percentiles
    logging.info("Calculating the 10th and 90th percentiles for meta description length.")
    tenth_percentile = merged_company_data['meta_description_length'].quantile(0.10)
    ninetieth_percentile = merged_company_data['meta_description_length'].quantile(0.90)
    logging.info(f"10th Percentile: {tenth_percentile}")
    logging.info(f"90th Percentile: {ninetieth_percentile}")

    logging.info("Flagging outliers in meta description length.")
    merged_company_data = flag_outliers(merged_company_data, ['meta_description_length'])

    # about 10% are flagged as outliers based on length
    outlier_counts = merged_company_data['meta_description_length_outlier'].value_counts()
    logging.info(f"Outlier counts by value:\n{outlier_counts}")

    # boxplot without outliers 
    logging.info("Plotting boxplot for meta description length without outliers.")
    meta_description_no_outliers  = merged_company_data[merged_company_data['meta_description_length_outlier'] == False]['meta_description_length']
    plot_helper.plot_boxplot(column=meta_description_no_outliers)

    # filter out len outliers, keep those between 10th - 90th Percentile
    logging.info("Filtering dataset to include only rows with meta description lengths between the 10th and 90th percentiles.")
    filtered_merged_company_data = merged_company_data[(merged_company_data['meta_description_length'] >= tenth_percentile) & (merged_company_data['meta_description_length'] <= ninetieth_percentile)]
    logging.info(f"Filtered meta description length statistics:\n{filtered_merged_company_data['meta_description_length'].describe()}")
    logging.info(f"Filtered dataset shape: {filtered_merged_company_data.shape}")

    # boxplot after removing outliers outside of 10th and 90th percentile
    logging.info("Plotting boxplot for filtered meta description length.")
    plot_helper.plot_boxplot(filtered_merged_company_data['meta_description_length'])

    # non-english meta descriptions
    logging.info("Analyzing the distribution of non-English meta descriptions.")
    logging.info(f"Meta description is English statistics:\n{filtered_merged_company_data['meta_description_is_english'].describe()}")
    plot_helper.plot_boxplot(filtered_merged_company_data['meta_description_is_english'])

    # EXPORT 
    non_eng_shape = filtered_merged_company_data[filtered_merged_company_data['meta_description_is_english'] < 0.35].shape
    logging.info(f"Exporting non-English meta descriptions. Shape of the data to export: {non_eng_shape}")
    filtered_merged_company_data[filtered_merged_company_data['meta_description_is_english'] < 0.35].sort_values(by='meta_description_is_english', ascending=False).to_csv(f'{file_outputs}/non_eng_meta_desc.csv', index=False)
    logging.info(f"Non-English meta descriptions successfully exported to {file_outputs}/non_eng_meta_desc.csv")

    # filter those over 35% 
    logging.info("Filtering out non-English meta descriptions with English probability lower than 35%.")
    filtered_merged_company_data = filtered_merged_company_data[filtered_merged_company_data['meta_description_is_english'] >= 0.35]

    # Step 1: Group by meta_description and count unique Categories in each group
    logging.info("Grouping by meta description to count unique categories.")
    category_counts = filtered_merged_company_data.groupby('meta_description')['Category'].nunique().reset_index()

    # Step 2: Rename the column for clarity
    category_counts = category_counts.rename(columns={'Category': 'unique_category_count'})

    # Step 3: Identify meta_descriptions to remove (where unique_category_count > 1)
    logging.info("Identifying meta descriptions that belong to more than one unique category.")
    meta_descriptions_to_remove = category_counts[category_counts['unique_category_count'] > 1]['meta_description']

    # export 
    logging.info(f"Exporting meta descriptions to remove (those associated with more than one category).")
    df = pd.DataFrame({'meta_description': meta_descriptions_to_remove})
    df.to_csv(f'{file_outputs}/duplicated_meta_descriptions_different_cat.csv', index=False)
    logging.info(f"Duplicated meta descriptions successfully exported to {file_outputs}/duplicated_meta_descriptions_different_cat.csv")

    # Step 4: Filter the original DataFrame to exclude these meta_descriptions
    initial_row_count = len(filtered_merged_company_data)
    logging.info(f"Filtering the original dataset to exclude the identified duplicated meta descriptions. Initial row count: {initial_row_count}")
    filtered_merged_company_data = filtered_merged_company_data[~filtered_merged_company_data['meta_description'].isin(meta_descriptions_to_remove)]
    final_row_count = len(filtered_merged_company_data)

    # Step 5: Count the total rows removed
    rows_removed = initial_row_count - final_row_count
    logging.info(f"Total rows removed after filtering: {rows_removed}")

    # handle remaining duplicates of the same category

    # Sort the DataFrame by 'meta_description'
    logging.info("Sorting the DataFrame by meta description.")
    filtered_merged_company_data = filtered_merged_company_data.sort_values(by='meta_description')

    # Count the initial number of rows
    initial_row_count = len(filtered_merged_company_data)

    # EXPORT
    logging.info("Exporting remaining duplicated meta descriptions within the same category.")
    filtered_merged_company_data[filtered_merged_company_data.duplicated(subset='meta_description', keep=False)].to_csv(f'{file_outputs}/duplicated_meta_descriptions_same_cat.csv', index=False)
    logging.info(f"Duplicated meta descriptions within the same category successfully exported to {file_outputs}/duplicated_meta_descriptions_same_cat.csv")

    # Identify duplicates based on 'meta_description' and keep only the first occurrence
    logging.info("Removing duplicates from the dataset, keeping only the first occurrence.")
    deduplicated_filtered_merged_company_data = filtered_merged_company_data[~filtered_merged_company_data.duplicated(subset='meta_description', keep='first')]

    # Count the final number of rows after deduplication
    final_row_count = len(deduplicated_filtered_merged_company_data)

    # Calculate the number of rows removed
    rows_removed = initial_row_count - final_row_count
    logging.info(f"Total rows removed after deduplication: {rows_removed}")

    # visualize distribution of meta_descriptions & meta_description_lengths per Category 
    logging.info("Starting visualization of meta descriptions and their lengths per category.")

    # Plot boxplot of meta description lengths per category
    logging.info("Plotting boxplot of meta description lengths per category.")
    plot_helper.plot_boxplot_meta_description_length(
        data=deduplicated_filtered_merged_company_data,
        category_col='Category',
        length_col='meta_description_length'
    )

    # Plot KDE of meta description lengths per category
    logging.info("Plotting KDE of meta description lengths per category.")
    plot_helper.plot_kde_meta_description_length(
        data=deduplicated_filtered_merged_company_data,
        category_col='Category',
        length_col='meta_description_length'
    )

    # Plot value counts of each category
    logging.info("Plotting value counts of each category.")
    plot_helper.plot_category_value_counts(
        data=deduplicated_filtered_merged_company_data,
        category_col='Category'
    )

    # Generate word clouds per category
    logging.info("Generating word clouds for each category.")
    plot_helper.generate_wordclouds_per_category(
        data=deduplicated_filtered_merged_company_data,
        category_col='Category',
        text_col='meta_description'
    )

    # Generate TF-IDF top terms per category
    logging.info("Generating TF-IDF top terms for each category.")
    tfidf_top_terms(
        data=deduplicated_filtered_merged_company_data,
        category_col='Category',
        text_col='meta_description',
        top_n=10  # Adjust this to get more or fewer top terms
    )
    logging.info("TF-IDF top terms generation complete.")

    logging.info(f"data for modelling shape: {deduplicated_filtered_merged_company_data.shape}")
    deduplicated_filtered_merged_company_data.to_csv(f'{data_modelling_path}/data_for_modelling.csv', index=False)
    logging.info(f"data for modelling saved to: {data_modelling_path}")
    
if __name__ == "__main__":
    main()