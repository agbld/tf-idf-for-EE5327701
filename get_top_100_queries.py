import os
import pandas as pd

def find_top_k_queries(folder_path, k):
    # List to store the number of rows and the corresponding file name
    file_row_counts = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Only process CSV files
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # Read the CSV file
            try:
                df = pd.read_csv(file_path)
            except:
                print(f'Error reading file: {file_path}')
                continue
            # Count the number of rows in the file
            row_count = len(df)
            # Extract the part after the underscore in the filename
            name_part = filename.split('_')[-1].replace('.csv', '')
            # Store the count and the extracted name part
            file_row_counts.append((row_count, name_part))

    # Sort the files by the number of rows in descending order
    file_row_counts.sort(reverse=True, key=lambda x: x[0])

    # Get the top K files
    top_k_files = file_row_counts[:k]

    # Create a DataFrame to store the results
    df_output = pd.DataFrame(top_k_files, columns=['count', 'name'])

    return df_output

if __name__ == '__main__':
    # Parameters
    folder_path = 'path_to_your_folder'  # Change this to your folder path
    k = 5  # Set the number of top files you want to extract
    output_file = 'top_k_files.csv'  # Output file name

    # Run the function
    top_k_queries_df = find_top_k_queries(folder_path, k, output_file)

    # Save the results to a CSV file
    top_k_queries_df.to_csv(output_file, index=False, encoding='utf-8')