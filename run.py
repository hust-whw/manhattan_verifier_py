from scipy.spatial.distance import cityblock
from sklearn.metrics import roc_curve
import pandas
import numpy as np
np.set_printoptions(suppress = True)

class ManhattanVerifier:
  
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
  
    def training(self):
        self.mean_vector = self.train.mean().values         
  
    def testing(self):
        # Mahattan distance
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, \
                                   self.mean_vector)
            self.user_scores.append(cur_score)
  
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, \
                                   self.mean_vector)
            self.imposter_scores.append(cur_score)

    def roc(self,t):
        # Classification
        labels = [0]*len(self.user_scores) + [1]*len(self.imposter_scores)
        scores = self.user_scores + self.imposter_scores
        
        # Compute FAR and FRR
        fars, tars, thresholds = roc_curve(labels, scores)
        frrs = 1 - tars
        i = np.where(thresholds == ([x for x in thresholds if x > t][-1] if [x for x in thresholds if x > t] else [x for x in thresholds][0]))
        j = np.where(thresholds == ([x for x in thresholds if x <= t][0] if [x for x in thresholds if x <= t] else [x for x in thresholds][-1]))
        far = fars[i[0][0]]
        frr = frrs[j[0][0]]

        # Compute EER
        dists = frrs - fars
        idx1 = np.argmin(dists[dists >= 0])
        idx2 = np.argmax(dists[dists < 0])
        x = [frrs[idx1], fars[idx1]]
        y = [frrs[idx2], fars[idx2]]
        eer = x[0] + (x[0] - x[1]) / (y[1] - x[1] - y[0] + x[0]) * (y[0] - x[0])

        # Keep 4 digit for float
        return round(far,4),round(frr,4),round(eer,4)
  
    def evaluate(self, threshold, type=None, user_index=1, N=200):
        ## Instruction of Type
        # 
        # KH => 'Key Hold'  KI => 'Key Interval'
        # The other inputs will be treated as the entire dataset
        cols_filter = [(x+1)*3 for x in range(11)]  \
            if type == 'KH'                         \
            else [(x+1)*3+1 for x in range(10)]     \
                if type == 'KI'                     \
                else [x+3 for x in range(31)]

        arr = []
        for subject in subjects:
            # Select key hold cols and Filter which user
            genuine_user_data = data \
                .iloc[:, cols_filter] \
                .loc[data.subject == subject, :]

            # Filter the other users
            imposter_data = data \
                .loc[data.subject != subject, :]

            # Set first N records to training set
            self.train = genuine_user_data[:N]

            # Set the other 400-N records to test set
            self.test_genuine = genuine_user_data[400-N:]

            # Set first 5 records per user as imposter
            self.test_imposter = imposter_data.groupby('subject').head(5) \
                .iloc[:, cols_filter]

            # Compute the template user
            self.training()

            # Compute the genuine and impostor scores using Manhattan distance
            self.testing()
            
            # Compute FAR and FRR for each user
            arr.append(self.roc(threshold))
        
        # if User index input is -1, output all users data 
        if user_index == -1:
            return arr
        else:
            return arr[user_index-1]

if __name__ == '__main__':
    path     = "/Users/hustwhw/Desktop/NYIT/CSCI-860/project/DSL-StrongPasswordData.csv"
    data = pandas.read_csv(path)
    subjects = data["subject"].unique()
    a = ManhattanVerifier(subjects)

    type = str(input("Please choose KeyHold or KeyInterval (KH or KI): ") or 'KH')
    user_index = int(input("Please choose User (1-51): ") or 1)
    N = int(input("Please input N: ") or 200)
    threshold = float(input("Please input Threshold: ") or 0.2)

    result = a.evaluate(threshold=float(threshold), type=str(type), user_index=int(user_index), N=int(N))

    if isinstance(result,list):
        # for i,x in enumerate(result):
        #     print("User "+str(i+1)+", FAR: "+str(x[0])+" FRR: "+str(x[1])+" EER: "+str(x[2]))
        print("Mean FAR: "+str(np.mean([x[0] for x in result])))
        print("Mean FRR: "+str(np.mean([x[1] for x in result])))
        print("Mean EER: "+str(np.mean([x[2] for x in result])))
    else:
        print("FAR: "+str(result[0]))
        print("FRR: "+str(result[1]))
        print("EER: "+str(result[2]))
