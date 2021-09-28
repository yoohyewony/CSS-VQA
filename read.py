import pickle
import os

ans2label_path = os.path.join('data', 'cache', 'trainval_ans2label.pkl')
ans2label = pickle.load(open(ans2label_path, 'rb'))
print(ans2label)
#print(f)