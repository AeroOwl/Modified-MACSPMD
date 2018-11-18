import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import pickle
import random
import os

"""
1. Withdraw: 

Existing relevance(<benign_API>) > threshold,

malseq1 -> benign_API -> malseq2

for i in range(1, limit):
    if i == 2:
        api_pattern = mal_API -> benign_API or benign_API -> mal_API
    
IT CANNOT GET RID OF BENIGN_API AND IT IS INSENSITIVE TO LONGER MALSEQ SINCE THE SERIES CONNNECTION !!!

Solution: Random walk is necessary.


2. Does consider API level make sense?
 
In PAM, all API of one patterns start from same level are regard as related. In other words, NO.


3. max_line = 46,000,000 | max_file_id = 13143


"""

class DataSet():

    def __init__(self):
        self.read_line = 5000
        self.filename = 'D:/TianChi/original/3rd_security_train/train.csv'
        self.file_label_dict = None

    def read_csv(self):
        return pd.read_csv(self.filename, sep=',', nrows=self.read_line)

    def parse_api_sequence(self):
        tid_dict = {}
        df = self.read_csv()
        for index, row in df.iterrows():
            # print(index)
            if len(row) == 5:
                file_id, api, tid, ret, order = row
            else:
                file_id, label, api, tid, ret, order = row
            # print(file_id, api, tid, order)
            if tid not in tid_dict:
                tid_dict[tid] = {}
                tid_dict[tid]["api_sequence"] = [api]
            else:
                tid_dict[tid]["api_sequence"].append(api)
            if len(row) != 5:
                tid_dict[tid]["label"] = label
            tid_dict[tid]["file_id"] = file_id
        return tid_dict

    # def print_api_sequence_pretty(self):
    #     tid_dict = self.parse_api_sequence()
    #     for tid in tid_dict:
    #         print("%s =>"%(tid))
    #         for s in tid_dict[tid]["api_sequence"]:
    #             print(s)
    #         print("- - - - - -\n")

    def get_all_file(self):
        file_set = set()
        data = pd.read_csv(self.filename, sep=',', usecols=[0], nrows=self.read_line)
        for index, row in data.iterrows():
            file_set.add(row[0])
        return file_set

    def get_all_label(self):
        label_set = set()
        data = pd.read_csv(self.filename, sep=',', usecols=[1], nrows=self.read_line)
        for index, row in data.iterrows():
            label_set.add(row[0])
        return label_set

    def get_all_api(self):
        api_set = set()
        data = pd.read_csv(self.filename, sep=',', usecols=[2], nrows=self.read_line)
        for index, row in data.iterrows():
            api_set.add(row[0])
        api_mapping = list(api_set)
        return api_mapping

    def get_all_tid(self):
        tid_set = set()
        data = pd.read_csv(self.filename, sep=',', usecols=[3], nrows=self.read_line)
        for index, row in data.iterrows():
            tid_set.add(row[0])
        return tid_set

    def get_all_file_and_its_label(self):
        file_dict = {}
        data = pd.read_csv(self.filename, sep=',', usecols=[0, 1], nrows=self.read_line)
        for index, row in data.iterrows():
            file_dict[row[0]] = row[1]
        return file_dict

    def get_freq(self, df, a, f):
        NS_a_f = 0
        NS_f = 0
        NA_a_f = 0
        NA_f = 0
        temp_file_id = -1
        isContain = False
        temp_file_set = set()
        for index, row in df.iterrows():
            file_id, label, api, tid, ret, order = row
            if a == api or f == label:
                if f == label:
                    NA_f += 1
                    if a == api:
                        NA_a_f += 1
                if file_id not in temp_file_set and f == label:
                    NS_f += 1
                    if a == api:
                        NS_a_f += 1
                        isContain = True
            if temp_file_id != file_id and isContain:
                temp_file_set.add(file_id)
                isContain = False
            temp_file_id = file_id
        # print(NA_f, NA_a_f, NS_f, NS_a_f)
        if NS_f == 0 or NA_f == 0:
            return 0.0
        return (NS_a_f * NA_a_f) / (NA_f * NS_f)

    def get_file_set(self):
        temp_file_set = set()
        df = self.read_csv()
        for index, row in df.iterrows():
            file_id, label, api, tid, ret, order = row
            temp_file_set.add(file_id)
        return temp_file_set

    def get_relevance(self, df, a, f, file_set):
        f = int(f)
        freq = self.get_freq(df, a, f)
        # print(a, f, freq)
        freq_sum = 0
        if freq < 0.00000001:
            return 0.0
        else:
            for file_id in file_set:
                freq_temp = self.get_freq(df, a, file_id)
                # print('~>', freq_temp, freq)
                freq_sum += freq_temp
            # temp for limitation
            if freq_sum == 0.0:
                return 0.0
            return freq / freq_sum

    def print_api_pretty(self, relv_dict):
        for f in relv_dict:
            print("{0} =>".format(f))
            top_api = sorted(relv_dict[f].items(), key=lambda s: s[1], reverse=True)
            for a in top_api:
                print(a[0], a[1])

    def trans_seq(self, mapping, api_list):
        if len(api_list) == 0:
            return "#"
        return "_".join(str(mapping.index(a)) for a in api_list if a in mapping)

    def parse_csv(self, chunk_size=10**6, total_size=10**7):
        api_set = set()
        label_set = set()
        file_set = set()
        tid_dict = {}
        file_label_dict = {}
        chunk_list = []
        for chunk in pd.read_csv(self.filename, chunksize=chunk_size, nrows=total_size):
            for index, row in chunk.iterrows():
                file_id, label, api, tid, ret, order = row
                api_set.add(api)
                label_set.add(label)
                file_set.add(file_id)
                if len(row) == 5:
                    file_id, api, tid, ret, order = row
                else:
                    file_id, label, api, tid, ret, order = row
                    file_label_dict[file_id] = label
                if tid not in tid_dict:
                    tid_dict[tid] = {}
                    tid_dict[tid]["api_sequence"] = [api]
                else:
                    tid_dict[tid]["api_sequence"].append(api)
                if len(row) != 5:
                    tid_dict[tid]["label"] = label
                tid_dict[tid]["file_id"] = file_id
            chunk_list.append(chunk)
        with open('tid_dict.pickle', 'wb') as f:
            pickle.dump(tid_dict, f)
        with open('file_label_dict.pickle', 'wb') as f:
            pickle.dump(file_label_dict, f)
        with open('api_set.pickle', 'wb') as f:
            pickle.dump(api_set, f)
        api_mapping = list(api_set)
        with open('api_mapping.pickle', 'wb') as f:
            pickle.dump(api_mapping, f)
        with open('label_set.pickle', 'wb') as f:
            pickle.dump(label_set, f)
        with open('file_set.pickle', 'wb') as f:
            pickle.dump(file_set, f)
        df = pd.concat(chunk_list)

        relv_dict = {}
        for f in label_set:
            f = int(f)
            for a in api_set:
                relevance = round(self.get_relevance(df, a, f, file_set), 4)
                # print("{0} ∈ {1} =>{2}".format(a, f, relevance))
                # if relevance > r:
                if f not in relv_dict:
                    relv_dict[f] = {}
                relv_dict[f][a] = relevance
        with open('relevance.pickle', 'wb') as f:
            pickle.dump(relv_dict, f)


    def get_timeseries(self):
        APIseq_list = []  # ... start by tid
        info_list = []
        with open('api_mapping.pickle', 'rb') as f:
            api_mapping = pickle.load(f)
        with open('tid_dict.pickle', 'rb') as f:
            tid_dict = pickle.load(f)
        for tid in tid_dict:
            important_APIseq = []
            important_APIseq_list = []
            file_id = tid_dict[tid]["file_id"]
            label = tid_dict[tid]["label"]
            APIseq_list.append(tid_dict[tid]["api_sequence"])
            info_list.append((file_id, label, tid))
            # for a in tid_dict[tid]["api_sequence"]:
            #     important_APIseq.append(a)
            # for span in range(1, n+1):
            #     for start in range(len(important_APIseq)-span):
            #         gram = important_APIseq[start:start+span]
            #         important_APIseq_list.append(gram)
            # for seq in important_APIseq_list:
            #     APIseq_list.append(self.trans_seq(api_mapping, seq))
            #     info_list.append((file_id, label, tid))

        # Merge the same API sequence
        # APISeq_dict = {}
        # for k in range(1, n+1):
        #     APISeq_dict[k] = {}
        #
        # for i in range(len(APIseq_list)):
        #     temp_k = len(APIseq_list[i].split('_'))
        #     if temp_k < n+1 and temp_k > 0:
        #         if APIseq_list[i] not in APISeq_dict[temp_k]:
        #             APISeq_dict[temp_k][APIseq_list[i]] = {}
        #             APISeq_dict[temp_k][APIseq_list[i]]["info"] = [info_list[i]]
        #         else:
        #             APISeq_dict[temp_k][APIseq_list[i]]["info"].append(info_list[i])
        return APIseq_list, info_list


    def get_relevant_subseq(self, r=0.69, n=3):
        APIseq_list = []  # ... start by tid
        info_list = []
        with open('api_mapping.pickle', 'rb') as f:
            api_mapping = pickle.load(f)
        with open('tid_dict.pickle', 'rb') as f:
            tid_dict = pickle.load(f)
        with open('relevance.pickle', 'rb') as f:
            relv_dict = pickle.load(f)
        for tid in tid_dict:
            important_APIseq = []
            important_APIseq_list = []
            file_id = tid_dict[tid]["file_id"]
            label = tid_dict[tid]["label"]

            ## all-in n-gram
            for a in tid_dict[tid]["api_sequence"]:
                if label in relv_dict and a in relv_dict[label] and relv_dict[label][a] > r:
                    important_APIseq.append(a)
            for span in range(1, n+1):
                for start in range(len(important_APIseq)-span):
                    gram = important_APIseq[start:start+span]
                    important_APIseq_list.append(gram)
            for seq in important_APIseq_list:
                APIseq_list.append(self.trans_seq(api_mapping, seq))
                APIseq_list.append((file_id, label, tid))

        # Merge the same API sequence
        APISeq_dict = {}
        for k in range(1, n+1):
            APISeq_dict[k] = {}

        for i in range(len(APIseq_list)):
            temp_k = len(APIseq_list[i].split('_'))
            if temp_k < n+1 and temp_k > 0:
                if APIseq_list[i] not in APISeq_dict[temp_k]:
                    APISeq_dict[temp_k][APIseq_list[i]] = {}
                    APISeq_dict[temp_k][APIseq_list[i]]["info"] = [info_list[i]]
                else:
                    APISeq_dict[temp_k][APIseq_list[i]]["info"].append(info_list[i])
        return APISeq_dict


    def generate_apiseq(self, n=3, r=0.69):
        self.parse_csv()
        return self.get_relevant_subseq(r=r, n=n)


    def raw_generate_apiseq(self, len_range, r=0.69):
        relv_dict = {}
        # df = self.read_csv()
        chunksize = 10 ** 6
        api_set = set()
        label_set = set()
        file_set = set()
        tid_dict = {}
        file_label_dict = {}
        chunk_list = []
        for chunk in pd.read_csv(self.filename, chunksize=chunksize):
            for index, row in chunk.iterrows():
                file_id, label, api, tid, ret, order = row
                api_set.add(api)
                label_set.add(label)
                file_set.add(file_id)
                if len(row) == 5:
                    file_id, api, tid, ret, order = row
                else:
                    file_id, label, api, tid, ret, order = row
                    file_label_dict[file_id] = label
                if tid not in tid_dict:
                    tid_dict[tid] = {}
                    tid_dict[tid]["api_sequence"] = [api]
                else:
                    tid_dict[tid]["api_sequence"].append(api)
                if len(row) != 5:
                    tid_dict[tid]["label"] = label
                tid_dict[tid]["file_id"] = file_id
            chunk_list.append(chunk)
            # print("api set: ", api_set)
            # print("label set: ", label_set)
            # print("file set: ", file_set)
        df = pd.concat(chunk_list)
        # api_set = self.get_all_api()
        # label_set = self.get_all_label()
        # file_set = self.get_file_set()
        for f in label_set:
            f = int(f)
            for a in api_set:
                relevance = round(self.get_relevance(df, a, f, file_set), 4)
                print("{0} ∈ {1} =>{2}".format(a, f, relevance))
                # if relevance > r:
                if f not in relv_dict:
                    relv_dict[f] = {}
                relv_dict[f][a] = relevance  # for retrieval
        # print(relv_dict)
        # self.print_APIs_pretty(relv_dict)
        APIseq_list = []  # ... start by tid
        info_list = []
        api_mapping = list(api_set)
        # tid_dict = self.parse_api_sequence()
        for tid in tid_dict:
            important_APIseq = []
            important_APIseq_list = []
            file_id = tid_dict[tid]["file_id"]
            label = tid_dict[tid]["label"]
            # print(tid_dict[tid])

            ## all-in
            for a in tid_dict[tid]["api_sequence"]:
                if label in relv_dict and a in relv_dict[label] and relv_dict[label][a] > r:
                    important_APIseq.append(a)

            for span in range(1, len_range+1):
                for start in range(len(important_APIseq)-span):
                    gram = important_APIseq[start:start+span]
                    # print(gram)
                    important_APIseq_list.append(gram)

            ## more detailed
            # for a in tid_dict[tid]["api_sequence"]:
            #     # print(a, file_id, type(file_id))
            #     if label in relv_dict and a in relv_dict[label] and relv_dict[label][a] > r:
            #         important_APIseq.append(a)
            #     elif len(important_APIseq) > 0:
            #         important_APIseq_list.append(important_APIseq)
            #         important_APIseq = []

            ## most detailed
            # str_seq_list = []
            # for seq in important_APIseq_list:
            #     str_seq_list.append(self.trans_seq(api_mapping, seq))
            # APIseq_list.append(str_seq_list)

            for seq in important_APIseq_list:
                APIseq_list.append(self.trans_seq(api_mapping, seq))
                info_list.append((file_id, label, tid))

        # APIseq_set = [list(t) for t in set(tuple(_) for _ in APIseq_list)]
        # APIseq_set.sort(key=APIseq_list.index)

        # Merge the same API sequence
        APISeq_dict = {}
        for k in range(1, len_range+1):
            APISeq_dict[k] = {}

        for i in range(len(APIseq_list)):
            temp_k = len(APIseq_list[i].split('_'))
            if temp_k < len_range and temp_k > 0:
                if APIseq_list[i] not in APISeq_dict[temp_k]:
                    APISeq_dict[temp_k][APIseq_list[i]] = {}
                    APISeq_dict[temp_k][APIseq_list[i]]["info"] = [info_list[i]]
                else:
                    APISeq_dict[temp_k][APIseq_list[i]]["info"].append(info_list[i])

        # for k in APISeq_dict:
        #     for i in APISeq_dict[k]:
        #         print(i, APISeq_dict[k][i])
        self.file_label_dict = file_label_dict

        return APISeq_dict, api_mapping, file_label_dict


class APISeqPattern():

    def __init__(self):
        self.count1 = 0
        self.count2 = 0
        self.support = 0
        self.confidence = 0
        self.ds = DataSet()
        # self.APISeq_set = self.ds.generate_APIseq()

    def malseq_pattern_mining(self, reference, C_k, k, DB, last_L, ms=0, mc=0):

        print("k => ", k)
        # reference_k = reference[k]

        APISeq_dict = C_k
        file_label_dict = DB

        count = 0
        for r in file_label_dict:
            if file_label_dict[r] != 0:
                count += 1

        for a in APISeq_dict:
            APISeq_dict[a]["count1"] = 0
            APISeq_dict[a]["count2"] = 0
            for occur in APISeq_dict[a]["info"]:
                if occur[0] in file_label_dict:
                    APISeq_dict[a]["count1"] += 1
                    if file_label_dict[occur[0]] != 0:
                        APISeq_dict[a]["count2"] += 1

        L = []
        MP = []
        for a in APISeq_dict:
            if count == 0:
                APISeq_dict[a]["support"] = 0.0
            else:
                APISeq_dict[a]["support"] = round(APISeq_dict[a]["count2"] / count, 6)
            # print(count, ms, APISeq_dict[a]["support"])
            if APISeq_dict[a]["support"] > ms:
                if APISeq_dict[a]["count1"] == 0:
                    APISeq_dict[a]["confidence"] = 0.0
                else:
                    APISeq_dict[a]["confidence"] = round(APISeq_dict[a]["count2"] / APISeq_dict[a]["count1"], 6)
                if k == 1:
                    L.append(a)
                    if APISeq_dict[a]["confidence"] > mc:
                        MP.append({"api_sequence": a, "details": APISeq_dict[a]})
                else:
                    c_dash = max([reference[k-1][x]["confidence"] for x in last_L])
                    # print("c': ", c_dash)
                    if APISeq_dict[a]["confidence"] > c_dash:
                        L.append(a)
                        if APISeq_dict[a]["confidence"] > mc:
                            MP.append({"api_sequence": a, "details": APISeq_dict[a]})

        # print("This is L =>")
        # for a in L:
        #     print(a, APISeq_dict[a]["info"], APISeq_dict[a]["support"], APISeq_dict[a]["confidence"])
        # print("This is MP =>")
        # for i in range(len(MP)):
        #     print(MP[i])

        C_k_new = self.generate_candidate(L, reference[k+1])

        return L, C_k_new, MP

    def generate_candidate(self, last_L, L):
        # print(reference)
        candidates = {}
        for longer_seq in L:
            seq_i = longer_seq.split("_")
            for seq in last_L:
                seq_j = seq.split("_")
                if set(seq_j).issubset(set(seq_i)):
                    candidates[longer_seq] = L[longer_seq]
        return candidates

    def mining_iteration(self, limit=5):
        result = []
        # total_APISeq_dict, api_mapping,  file_label_dict = self.ds.generate_apiseq(len_range=limit)
        # file_label_dict = self.ds.get_all_file_and_its_label()
        total_APISeq_dict = self.ds.generate_apiseq(n=3, r=0.69)
        with open('file_label_dict.pickle', 'rb') as f:
            file_label_dict = pickle.load(f)
        temp_L = {}
        C_k_new = total_APISeq_dict[1]
        for k in range(1, limit):
            temp_L, C_k_new, MP = self.malseq_pattern_mining(reference=total_APISeq_dict, C_k=C_k_new, k=k, DB=file_label_dict, last_L=temp_L)
            result.append(MP)
        return result

    def measurement(self, mining_result):
        benign_cnt = 0
        malicious_cnt = 0
        file_label_dict = self.ds.get_all_file_and_its_label()
        for i in file_label_dict:
            if file_label_dict[i] == 0:
                benign_cnt += 1
            else:
                malicious_cnt += 1
        # sequence coverage of benign or malicious files
        benign_set = set()
        malicious_set = set()
        for records in mining_result:
            for r in records:
                for occur in r["details"]["info"]:
                    file_id = occur[0]
                    label = occur[1]
                    if label == 0:
                        benign_set.add(file_id)
                    else:
                        malicious_set.add(file_id)
        print("Coverage => benign: {0} / {1} | malicious: {2} / {3}".format(len(benign_set), benign_cnt, len(malicious_set), malicious_cnt))


class VecSmith():

    def seq2vec(self, seqs_info, seqs_mapping=None):
        files = set()
        seqs = set()
        for records in seqs_info:
            for r in records:
                seqs.add(r["api_sequence"])
                for occur in r["details"]["info"]:
                    file_id = occur[0]
                    files.add(file_id)
        files_mapping = list(files)
        if seqs_mapping == None:
            seqs_mapping = list(seqs)
        matrix = np.zeros([len(files_mapping), len(seqs_mapping)])

        for records in seqs_info:
            for r in records:
                row_idx = seqs_mapping.index(r["api_sequence"])
                for occur in r["details"]["info"]:
                    file_id = occur[0]
                    label = occur[1]
                    col_idx = files_mapping.index(file_id)
                    if label != 0:
                        matrix[col_idx][row_idx] = 1
                    # matrix[-1][row_idx] = label
        df = pd.DataFrame(matrix, columns=seqs_mapping, index=files_mapping)
        return df, seqs_mapping

    def get_features(self, features=None):
        miner = APISeqPattern()
        seqs = miner.mining_iteration()
        dataframe, features = self.seq2vec(seqs, seqs_mapping=features)
        print(dataframe)
        return dataframe, features

    def dataset_split(self, data):
        train_data, test_data = train_test_split(data, test_size=0.3, random_state=10)
        return train_data, test_data


class MalwareDetector():

    def __init__(self):
        self.model = None
        self.features = None

    def train(self, train_data, labels):
        clf = RandomForestClassifier(n_jobs=5, random_state=0)
        clf.fit(train_data, labels)
        with open('clf.pickle', 'wb') as f:
            pickle.dump(clf, f)
        self.model = clf
        return clf

    def predict(self, test_data):
        return self.model.predict(test_data)
        # return self.model.predict_proba(test_data)

    def store_features(self, features):
        self.features = features
        with open('features.pickle', 'wb') as f:
            pickle.dump(features, f)

    def fetch_features(self):
        with open('features.pickle', 'rb') as f:
            features = pickle.load(f)
        return features


if __name__ == "__main__":

    ds = DataSet()
    vs = VecSmith()

    dataframe, features = vs.get_features()
    train, test = vs.dataset_split(dataframe)
    print("train =>", train)
    print("test  =>", test)

    labels = []
    for index, row in train.iterrows():
        file_id = row[0]
        labels.append(ds.file_label_dict[int(file_id)])
    # print(labels)

    # validation
    y_true = []
    for index, row in test.iterrows():
        file_id = row[0]
        y_true.append(ds.file_label_dict[int(file_id)])

    md = MalwareDetector()
    md.store_features(features)
    md.train(train, labels)
    y_pred = md.predict(test)
    print(y_true, y_pred)

    print(f1_score(y_true, y_pred, average="macro"))
    print(precision_score(y_true, y_pred, average="macro"))
    print(recall_score(y_true, y_pred, average="macro"))
    # print(classification_report(y_true, y_pred, labels=labels))

    # TEST
    print("Start testing ...")

    with open('clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)

    features = md.fetch_features()

    test_file = '../original/3rd_security_test/test.csv'
    # test_df, features = vs.get_features(filename=test_file, features=features)  # column error
    test_ds = DataSet()
    test_ds.filename = test_file
    tid_seqs = test_ds.parse_api_sequence()
    file_seqs = {}
    for tid in tid_seqs:
        file_id = tid_seqs[tid]["file_id"]
        if file_id not in file_seqs:
            file_seqs[file_id] = [tid_seqs[tid]["api_sequence"]]
        else:
            file_seqs[file_id].append(tid_seqs[tid]["api_sequence"])
    for f in file_seqs:
        print(file_seqs[f])

    len_range = 5
    test_files_mapping = []
    for f in file_seqs:
        test_files_mapping.append(f)
    test_matrix = np.zeros([len(test_files_mapping), len(features)])

    with open('api_mapping.pickle', 'rb') as f:
        api_mapping = pickle.load(f)

    for f in file_seqs:
        seq_list = []
        for seq in file_seqs[f]:
            for span in range(1, min(len_range, len(seq))+1):
                for start in range(len(seq)-span):
                    gram = seq[start:start+span]
                    # print(gram)
                    seq_list.append(test_ds.trans_seq(api_mapping, gram))
        for i in range(len(features)):
            col_idx = test_files_mapping.index(f)
            if features[i] in seq_list:
                test_matrix[col_idx][i] = 1
            else:
                test_matrix[col_idx][i] = 0
    test_df = pd.DataFrame(test_matrix, columns=features, index=test_files_mapping)
    print("test data: ", test_df)
    # test_pred = md.predict(test_df)
    test_pred = clf2.predict_proba(test_df)
    print(test_pred)
