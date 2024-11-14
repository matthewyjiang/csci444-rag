#!/bin/bash

# Define the path to the text file containing arguments
ARG_FILE="questions.txt"

# Define the Python script to run
PYTHON_SCRIPT="query_data.py"

# Define the output file for results
RESULT_FILE="results.txt"

counter=1

# Loop through each line in the text file
while IFS= read -r arg; do
  echo "Question $counter: $arg" >> "$RESULT_FILE"

  # Run the Python script with the argument
  python3 "$PYTHON_SCRIPT" "$arg" >> "$RESULT_FILE"
  echo "-----------------------------------" >> "$RESULT_FILE"
  ((counter++))
done < "$ARG_FILE"



