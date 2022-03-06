import numpy as np
from sklearn.preprocessing import LabelBinarizer

def save_embeddings(embeddings, path):
    with open(path, 'w', encoding="utf-8") as file:
        # print(len(embeddings))
        for index, embedding in enumerate(embeddings):
            # print(len(embedding))
            file.write(str(index) + ' ')
            for i in range(len(embedding)):
                file.write(str(embedding[i]) + ' ')
            file.write('\n')

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

def label_sum(label_train):
    label_sum=0
    for each in label_train:
        label_sum=label_sum+each
    return label_sum

def average_value(list):
    return float(sum(list))/len(list)

