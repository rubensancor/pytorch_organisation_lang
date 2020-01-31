#! /bin/sh

datasets="noVerizon Hashtag HashtagMention HashtagUrl Mention MentionUrl nothing Url"

for dataset in $datasets; do
    python3 main.py -f noVerizon/mixed_"$dataset".csv -e 10 -b 4096 -ws
    if [ "$?" = "0" ]; then
        exit 0
    else
    echo "GPU Memory error, changing to MIXED MODE"
	python3 main.py -f noVerizon/mixed_"$dataset".csv -e 10 -b 4096 -ws --mixed_memory 
    fi
done