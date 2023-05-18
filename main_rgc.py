import os
import csv
if __name__ ==  '__main__' :
    ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/' 
    DATA_NAME =  'CD_5' 
    DATA_PATH = ROOT +  'dataset/{}/{}.json' .format(DATA_NAME, DATA_NAME)
    LOG_NAME = 'log_rgc'


    dropout = [0.6, 0.7, 0.8, 0.9]
    beta = [0.3, 0.6, 1.0, 1.4]

    for beta_rate in beta:
        for drop_rate in dropout:
            for i in range(5):
                os.system("python %srgc.py \
                          --root_path %s \
                          --dataset_name %s \
                          --dataset_path %s \
                          --logger logger \
                          --logger_file %slog/btdo%s.log \
                          --gcn_dropout %0.1f \
                          --train_max_iter 1000 \
                          --beta %0.1f \
                          --train_early_stopping_patience 80 \
                          --device 2\
                          --train_classification False"\
                          %(ROOT, ROOT, DATA_NAME, DATA_PATH, ROOT, (str(beta_rate)+str(drop_rate)), drop_rate, beta_rate))
