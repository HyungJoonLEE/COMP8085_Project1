from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scripts import preprocess as ref
import pandas as pd


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

origin = pd.read_csv(ORIGINAL_CSV, encoding='ISO-8859-1', low_memory=False)
df = ref.preprocess_data(origin)


x = df.drop(['attack_cat', 'Label'], axis=1)
y = df['Label']

# Train model with 30% of data will be used as a test model
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=42)


# Default: max_depth=3, learning_rate=0.1
gbc = GradientBoostingClassifier(random_state=0,
                                 max_depth=3,
                                 learning_rate=0.1)
gbc.fit(x_train, y_train)

# train set accuracy
score_train = gbc.score(x_train, y_train)
print('{:.3f}'.format(score_train))

# Generalization accuracy
score_test = gbc.score(x_test, y_test)
print('{:.3f}'.format(score_test))

# Visualize the result
n_feature = x.shape[1]
index = np.arange(n_feature)
plt.barh(index, gbc.feature_importances_, align='center')
plt.yticks(index, df.feature_names)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)
plt.show()
