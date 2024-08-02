#!/bin/bash

# Define the output file
OUTPUT_FILE="python_files_info.txt"

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Find all .py files in the current directory and subdirectories
find . -type f -name "*.py" | while read -r file; do
    # Append the file path to the output file
    echo "File path: $file" >> "$OUTPUT_FILE"
    
    # Append the file content to the output file
    echo "Content:" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    
    # Add a separator for readability
    echo -e "\n---\n" >> "$OUTPUT_FILE"
done

echo "Information saved to $OUTPUT_FILE"