#! /bin/bash

datasets="noVerizon Hashtag HashtagMention HashtagUrl Mention MentionUrl nothing Url"

for dataset in $datasets; do
    python3 main.py -f noVerizon/mixed_"$dataset".csv --seed 1234 -ws
    if [ "$?" = "0" ]; then
        :
    else
        echo "GPU Memory error, changing to MIXED MODE"
        python3 main.py -f noVerizon/mixed_"$dataset".csv --seed 1234 -ws --mixed_memory 
    fi
done
