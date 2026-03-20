# 
# sync datasets to remote server
# usage: bash scripts/rsync_dataset.sh

rsync -avzP datasets/ robby@sally-server:/home/robby/workspace/Oil-Spill-Benz/datasets

rsync -avzP datasets/lados_432 robby@sally-server:/home/robby/workspace/Oil-Spill-Benz/datasets/



# Move files (default)
python scripts/separate_by_date.py datasets/raw/dv3-combined datasets/processed/dv3-by-date

# Copy files instead
python scripts/separate_by_date.py datasets/raw/dv3-combined datasets/processed/dv3-by-date --copy