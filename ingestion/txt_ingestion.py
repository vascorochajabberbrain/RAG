def txt_lines_as_an_array(filename):
    """
    Reads a text file and returns a list of non-empty lines as strings.
    
    Args:
        filename (str): Path to the input .txt file
        
    Returns:
        list: List of strings, each being a non-empty line from the file
    """
    non_empty_lines = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, start=1):
                stripped = line.strip()
                if stripped:  # Only add if line has content
                    non_empty_lines.append(stripped)
                # Optional: uncomment to debug empty lines
                # else:
                #     print(f"Skipped empty line {line_num}")
        
        return non_empty_lines
    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return []
    except PermissionError:
        print(f"Error: Permission denied when accessing '{filename}'.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []



def main():
    # Replace with your actual file path
    file_path = 'ingestion/data_to_ingest/txt_files/informacao_nutricional.txt'
    
    lines = txt_lines_as_an_array(file_path)
    
    if lines:
        print(f"Successfully loaded {len(lines)} non-empty lines:\n")
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line}")
    else:
        print("No lines were loaded. Check the file path and content.")

    return lines


if __name__ == "__main__":
    main()

