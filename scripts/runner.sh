python run_experiments.py --config="configs/experiments/exp2_2_port-oil.yaml" | tee logs/output.txt 
python run_experiments.py --config="configs/experiments/exp2_1_port-oil.yaml" | tee logs/output-port-oil.txt 


python run_experiments.py --config="configs/experiments/exp2_4_dv3train_dryrun.yaml" | tee -a logs/output-exp2_4_dv3train.txt 

