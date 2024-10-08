import matplotlib.pyplot as plt
import seaborn as sns
import os
from wordcloud import WordCloud

script_dir = os.path.dirname(__file__)
plot_file_path = os.path.join(script_dir, '../../../data/plots')

def generate_wordclouds_per_category(data, category_col, text_col):
    """
    Generates and displays a word cloud for each category.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    category_col (str): The name of the column containing the categories.
    text_col (str): The name of the column containing the text for word cloud generation.

    Returns:
    None
    """
    categories = data[category_col].unique()

    for category in categories:
        # Filter the data for the current category
        category_data = data[data[category_col] == category]
        
        # Combine all text from the text column into one string
        combined_text = ' '.join(category_data[text_col].dropna())

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Word Cloud for Category: {category}')
        plt.axis('off')
        
         # Save the plot 
        plot_filename = os.path.join(plot_file_path, f'{category}_word_cloud.png')
        plt.savefig(plot_filename)
        
def plot_token_length_distribution(token_lengths, bins=50):
    """
    Analyze and plot the distribution of token lengths in the dataset.

    Args:
    - texts (pd.Series or list): The dataset containing the text samples.
    - tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
    - add_special_tokens (bool): Whether to add special tokens (e.g., [CLS], [SEP]) (default is True).
    - bins (int): Number of bins for the histogram (default is 50).

    Returns:
    - token_lengths (list): A list containing the length of tokens for each text.
    """
    # Plot the distribution of token lengths
    plt.hist(token_lengths, bins=bins)
    plt.title('Distribution of Token Lengths')
    plt.xlabel('Token Length')
    plt.ylabel('Number of Texts')
    # Save the plot 
    plot_filename = os.path.join(plot_file_path, 'token_length_distribution.png')
    plt.savefig(plot_filename)
    
def plot_confusion_matrix(conf_matrix, figsize=(8, 6), cmap='Blues'):
    """
    Plot and display the confusion matrix as a heatmap.

    Args:
    - conf_matrix (array-like): The confusion matrix to be plotted.
    - figsize (tuple): The size of the figure (default is (8, 6)).
    - cmap (str): The color map to use for the heatmap (default is 'Blues').

    Returns:
    - None
    """
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Save the plot 
    plot_filename = os.path.join(plot_file_path, 'confusion_matrix.png')
    plt.savefig(plot_filename)
        
def plot_category_value_counts(data, category_col):
    """
    Plots the value counts of each category in a bar chart and saves the plot to a file.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    category_col (str): The name of the column containing the categories.
    output_path (str): The directory where the plot should be saved.

    Returns:
    None
    """
    # Get value counts for each category
    value_counts = data[category_col].value_counts()

    # Plot the value counts
    plt.figure(figsize=(12, 8))
    sns.barplot(x=value_counts.index, y=value_counts.values, palette="viridis")
    plt.title('Value Counts of Each Category')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # save plot
    plot_filename = os.path.join(plot_file_path, f'{category_col}_value_counts.png')
    plt.savefig(plot_filename)
    
     

def plot_boxplot_meta_description_length(data, category_col, length_col):
    """
    Plots a boxplot for the distribution of meta_description lengths per category
    and saves the plot to a file.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    category_col (str): The name of the column containing the categories.
    length_col (str): The name of the column containing the meta description lengths.
    output_path (str): The directory where the plot should be saved.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=category_col, y=length_col, data=data)
    plt.title('Distribution of Meta Description Lengths per Category')
    plt.xlabel('Category')
    plt.ylabel('Meta Description Length (in words)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot 
    plot_filename = os.path.join(plot_file_path, f'{category_col}_vs_{length_col}_boxplot.png')
    plt.savefig(plot_filename)
    


def plot_kde_meta_description_length(data, category_col, length_col):
    """
    Plots a KDE plot for the distribution of meta_description lengths per category
    and saves the plot to a file.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    category_col (str): The name of the column containing the categories.
    length_col (str): The name of the column containing the meta description lengths.
    output_path (str): The directory where the plot should be saved.

    Returns:
    None
    """
    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=data, x=length_col, hue=category_col, fill=True, common_norm=False, alpha=0.5)
    plt.title('KDE Plot of Meta Description Lengths per Category')
    plt.xlabel('Meta Description Length (in words)')
    plt.ylabel('Density')

    # save plot
    plot_filename = os.path.join(plot_file_path, f'{category_col}_vs_{length_col}_kdeplot.png')
    plt.savefig(plot_filename)
    
     

def plot_boxplot(column):
    """
    Plots and displays a boxplot for a given Pandas column and saves the plot to a file.

    Parameters:
    column (pd.Series): The Pandas Series (column) for which to plot the boxplot.
    output_path (str): The directory where the plot should be saved.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.boxplot(column.dropna(), vert=False, patch_artist=True)
    plt.title(f'Boxplot of {column.name}')
    plt.xlabel(column.name)
    plt.grid(True)

    # Save the plot 
    plot_filename = os.path.join(plot_file_path, f'{column.name}_boxplot.png')
    plt.savefig(plot_filename)
    
     