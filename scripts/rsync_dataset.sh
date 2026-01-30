# 
# sync datasets to remote server
# usage: bash scripts/rsync_dataset.sh

rsync -avzP datasets/ robby@sally-server:/home/robby/workspace/Oil-Spill-Benz/datasets
