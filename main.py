import os
import csv
# beta 1

if __name__ ==  '__main__' :
    ROOT =  '/home/yeonghwa/workspace/reviewgraph/final_cd/'
    DATA_NAME =  'CD_5' 
    DATA_PATH = ROOT +  'dataset/{}/{}.json' .format(DATA_NAME, DATA_NAME)
    LOG_NAME = 'log_final'
    

#     ed = [0.2, 0.8]
#     nd = [0.2, 0.8, 1.0]
    ed = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    nd = [0.2, 0.4, 0.6, 0.8, 1.0]
    beta = [0.0, 1.0]

#     for beta_rate in beta:
#         with open(ROOT+LOG_NAME+'/best.csv', 'a', newline='') as f:
#             w = csv.writer(f)
#             w.writerow([beta_rate])
#             os.system("python %s/rgc.py \
#                               --root_path %s \
#                               --dataset_name %s \
#                               --dataset_path %s \
#                               --logger logger \
#                               --logger_file %slog_rgc/bt%s.log \
#                               --gcn_dropout 0.7 \
#                               --train_early_stopping_patience 80 \
#                               --train_max_iter 1000 \
#                               --beta %0.1f \
#                               --train_classification False"\
#                               %(ROOT, ROOT, DATA_NAME, DATA_PATH, ROOT, beta_rate , beta_rate))
    for beta_rate in beta:
        for nd_rate in nd:
            for ed_rate in ed:
                with open(ROOT+LOG_NAME+'/best.csv', 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([beta_rate,ed_rate, nd_rate])
                    os.system("python %s/rgc_nd_ed.py \
                    --root_path %s \
                    --dataset_name %s \
                    --dataset_path %s \
                    --logger logger \
                    --logger_file %slog_final/btednd%s.log \
                    --gcn_dropout 0.7 \
                    --ed_alpha %0.1f \
                    --nd_alpha %0.1f \
                    --train_early_stopping_patience 80 \
                    --train_max_iter 1000 \
                    --device 1\
                    --beta %0.1f \
                    --train_classification False"\
                              %(ROOT, ROOT, DATA_NAME, DATA_PATH, ROOT, (str(beta_rate)+str(ed_rate)+str(nd_rate)), ed_rate, nd_rate, beta_rate))
    
    
    
    