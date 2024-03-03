import pandas as pd
from scripts import preprocess as ref

ORIGINAL_CSV = '../data/UNSW-NB15-BALANCED-TRAIN.csv'

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


def main():
    # Read csv using pandas in Latin mode
    origin = pd.read_csv(ORIGINAL_CSV, encoding='ISO-8859-1', low_memory=False)
    df = ref.preprocess_data(origin)

    print(df.info())

    return 0


if __name__ == "__main__":
    main()
