import sys
import os
import argparse

import KNN
import pandas as pd
from scripts import preprocess as ref
from sklearn.model_selection import train_test_split
from models import RFE
from models import KNN
from models import GBC
from models import CCA


ORIGINAL_CSV = './data/UNSW-NB15-BALANCED-TRAIN.csv'

features = ['srcip', 'sport', 'dstip', 'dsport',
            'proto', 'state', 'dur', 'sbytes',
            'dbytes', 'sttl', 'dttl', 'sloss',
            'dloss', 'service',	'Sload', 'Dload',
            'Spkts', 'Dpkts', 'swin', 'dwin',
            'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
            'Stime', 'Ltime', 'Sintpkt', 'Dintpkt',	'tcprtt',
            'synack', 'ackdat',	'is_sm_ips_ports', 'ct_state_ttl',
            'ct_flw_http_mthd',	'is_ftp_login',	'ct_ftp_cmd', 'ct_srv_src',
            'ct_srv_dst', 'ct_dst_ltm',	'ct_src_ ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm',	'ct_dst_src_ltm', 'attack_cat',	'Label']


def get_args(args):
    parser = argparse.ArgumentParser(description="COMP8085 Project 1")
    task_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('test_set',
                        type=str,
                        help='Path to held-out test set, in CSV format')
    parser.add_argument('classification_method',
                        type=str,
                        help='Selected classification method name')
    task_group.add_argument('--Label',
                            dest='task',
                            action='store_const',
                            const='Label',
                            help='Predict whether the packet is normal or not')
    task_group.add_argument('--attack_cat',
                            dest='task',
                            action='store_const',
                            const='attack_cat',
                            help='Predict the attack category')
    return parser


function_hashmap = {
    'RFE': RFE.rfe_model,
    'KNN': KNN.knn_model,
    'CCA': CCA.correlation_coefficient
}


def run_function_by_key(key, training_data, test_data, validation_data, target):
    if key in function_hashmap:
        function_to_run = function_hashmap[key]
        # TODO: Need to find out what will be the arguments
        #       for our classifier
        function_to_run(training_data, test_data, validation_data,target)
    else:
        print(f"No function found for key: {key}")


def main():
    global task_
    print(os.getcwd())
    args = get_args(sys.argv[1:]).parse_args()

    # Read csv using pandas in Latin mode
    original_csv = pd.read_csv(ORIGINAL_CSV,
                               encoding='ISO-8859-1',
                               low_memory=False)

    # Process data - change values of ports and null + factorize
    df = ref.preprocess_data(original_csv)

    # Split original csv into train and validate+test (0.7 : 0.3)
    train_df, validate_test_df = train_test_split(df,
                                                  train_size=0.7,
                                                  shuffle=True,
                                                  stratify=df[args.task],
                                                  random_state=32)

    # Split validate+test into validate and test (0.5 : 0.5)
    validate_df, test_df = train_test_split(validate_test_df,
                                            train_size=0.5,
                                            shuffle=True,
                                            stratify=validate_test_df[args.task],
                                            random_state=34)

    # Save in to csv format
    train_df.to_csv('./data/train_data.csv', index=False)
    test_df.to_csv('./data/test_data.csv', index=False)
    validate_df.to_csv('./data/validate_data.csv', index=False)

    # print(df.shape)
    # print(train_df.shape)
    # print(validate_df.shape)
    # print(test_df.shape)

    #facotrize scrip and dstip
    # Run
    run_function_by_key(args.classification_method,
                        train_df,
                        test_df,
                        validate_df,
                        args.task)


    # print(df.info())

    return 0


if __name__ == "__main__":
    main()
