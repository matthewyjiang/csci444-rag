# clean_glove.py

import sys

def clean_glove_file(input_path: str, output_path: str, invalid_char: str = '�'):
    """
    Removes lines containing the specified invalid character from the GloVe file.

    Parameters:
    - input_path (str): Path to the original GloVe file.
    - output_path (str): Path to save the cleaned GloVe file.
    - invalid_char (str): The character indicating invalid/malformed lines.
    """
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, 1):
            if invalid_char in line:
                print(f"⚠️ Skipping line {line_number} due to invalid character.")
                continue
            outfile.write(line)
    
    print(f"✅ Cleaned GloVe file saved to '{output_path}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 clean_glove.py <input_glove_path> <output_clean_glove_path>")
        sys.exit(1)
    
    input_glove = sys.argv[1]
    output_clean_glove = sys.argv[2]
    
    clean_glove_file(input_glove, output_clean_glove)