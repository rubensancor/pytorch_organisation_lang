#! /bin/bash

# datasets="full hashtag hashtagMention hashtagUrl mention mentionUrl nothing url"
datasets="hashtag"



for dataset in $datasets; do
    for i in {1..5}; do
        # python3 main.py -f "$dataset" -b 1450 -ws --users
        python3 main.py -f "$dataset" -e 10 -b 1450 --lr 0.001816 --dropout 0.5374 --dense1_size 849 --dense2_size 282 --kernel_start 8 --kernel_steps 2 -ws --users
        if [ "$?" = "0" ]; then
            :
        else
            echo "GPU Memory error, changing to MIXED MODE"
            python3 main.py -f "$dataset" -b 1450 -ws --users --mixed_memory 
        fi
    done
done