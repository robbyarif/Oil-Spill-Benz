python run_experiments.py --config="configs/experiments/exp2_2_port-oil.yaml" | tee logs/output-2_2_port-oil.txt 
python run_experiments.py --config="configs/experiments/exp2_1_port-oil.yaml" | tee logs/output-port-oil.txt 


python run_experiments.py --config="configs/experiments/exp2_4_dv3train_dryrun.yaml" | tee -a logs/output-exp2_4_dv3train.txt 
python run_experiments.py --config="configs/experiments/exp2_4_dv3train.yaml" | tee -a logs/output-exp2_4_dv3train-20260206.txt 

python run_experiments.py --config="configs/experiments/exp2.3a_port-oil-pretrain.yaml" | tee -a logs/output-exp2.3a_port-oil-pretrain-20260223.txt 
python run_experiments.py --config="configs/experiments/exp4_dv3-dv4_crop.yaml" | tee -a logs/output-exp4_dv3-dv4_crop-20260328.txt 
