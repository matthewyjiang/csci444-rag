#!/bin/bash

if [ "$1" == "467" ] || [ "$1" == "444" ]; then
    data_group="$1"
else
    echo "Invalid argument. Please provide '467' or '444' as the data_group."
    exit 1
fi

if [ "$2" == "tfidf" ] || [ "$2" == "nomic" ]; then
    method="$2"
else
    echo "Invalid argument. Please provide 'tfidf' or 'nomic' as the method."
    exit 1
fi

data_path="${data_group}data"
questions_path="${data_group}questions"

# Run the python script

python3 populate_database.py --reset $method $data_path

python3 validate_responses.py $questions_path



