nominal_ = ['srcip', 'dstip', 'proto', 'state',
            'service', 'attack_cat']

integer_ = ['sport', 'dsport', 'sbytes', 'dbytes',
            'sttl', 'dttl', 'sloss', 'dloss',
            'Spkts', 'Dpkts', 'swin', 'dwin',
            'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'ct_state_ttl',
            'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src',
            'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm',
            'ct_dst_src_ltm']

float_ = ['dur', 'Sload', 'Dload', 'Sjit',
          'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt',
          'synack', 'ackdat']

tstamp_ = ['Stime', 'Ltime']

binary_ = ['is_sm_ips_ports', 'is_ftp_login', 'Label']

factorize_ = ['proto', 'state', 'service', 'attack_cat']

object_ = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state',
                'service', 'ct_ftp_cmd', 'attack_cat']

ip_ = ['srcip', 'dstip']
port_ = ['sport', 'dsport']

