#! /bin/bash

# datasets="full hashtagMention hashtagUrl mention mentionUrl url"
datasets="hashtagMention"

for dataset in $datasets; do
    for i in {1..5}; do
        python3 main.py -f "$dataset" -b 1450 -ws --users
        # python3 main.py -f noVerizon/mixed_"$dataset".csv -e 10 -b 2048 --lr 0.001816 --dropout 0.5374 --dense1_size 849 --dense2_size 282 --kernel_start 8 --kernel_steps 2 -ws
        if [ "$?" = "0" ]; then
            :
        else
            echo "GPU Memory error, changing to MIXED MODE"
            python3 main.py -f "$dataset" -b 1450 -ws --users --mixed_memory 
        fi
    done
done