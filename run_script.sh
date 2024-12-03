#!/bin/bash

if [ "$1" == "467" ] || [ "$1" == "444" ]; then
    data_group="$1"
else
    echo "Invalid argument. Please provide '467' or '444' as the data_group."
    exit 1
fi

data_path="${data_group}data"
questions_path="${data_group}questions"

# Run the python script

python3 populate_database.py --reset $data_path --embedding_path word2vec_embedding.txt

python3 validate_responses.py $questions_path --embedding_path word2vec_embedding.txt




