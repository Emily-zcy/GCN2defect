import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
import tensorflow as tf
from sklearn import model_selection, metrics
from utils import MyLabelBinarizer, save_embeddings
from IPython.display import display, HTML

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.Data Preparation, Loading the Ant network
dataset = sg.datasets.Ant()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

print(G.info())
node_subjects.value_counts().to_frame()
print(node_subjects.value_counts().to_frame())

# Splitting the data：8：1：1
train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=0.8, test_size=None, stratify=node_subjects, random_state=0
)
val_subjects, test_subjects = model_selection.train_test_split(
    test_subjects, train_size=0.5, test_size=None, stratify=test_subjects, random_state=0
)
train_subjects.value_counts().to_frame()
print(train_subjects.value_counts().to_frame())

# Converting to numeric arrays
# LabelBinarizer cannot convert binary tags into one-hot encoded form,
# Encapsulate a Labelbinarizer function: MyLabelBinarizer
target_encoding = MyLabelBinarizer()

train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

# 2. Creating the GCN layers
generator = FullBatchNodeGenerator(G, method="gcn")
train_gen = generator.flow(train_subjects.index, train_targets)
gcn = GCN(
    layer_sizes=[32, 32], activations=["relu", "relu"], generator=generator, dropout=0.1
)
x_inp, x_out = gcn.in_out_tensors()
predictions = tf.keras.layers.Dense(units=train_targets.shape[1],
                                    activation="softmax")(x_out)

# 3. Training and evaluating
model = tf.keras.Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['acc']
)
val_gen = generator.flow(val_subjects.index, val_targets)

es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=200, restore_best_weights=True)
history = model.fit(
    train_gen.inputs,
    train_gen.targets,
    epochs=2000,
    validation_data=(val_gen.inputs, val_gen.targets),
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    # callbacks=[es_callback],
)

sg.utils.plot_history(history)

test_gen = generator.flow(test_subjects.index, test_targets)
test_metrics = model.evaluate(test_gen.inputs, test_gen.targets)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# Making predictions with the model
all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen.inputs)
# use the inverse_transform method of our target attribute specification to turn these values back to the original categories
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
# Let’s have a look at a few predictions after training the model
df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
df.head(20)

# Node embeddings
embedding_model = tf.keras.Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(all_gen.inputs)
print(emb.shape)
X = emb.squeeze(0)
print(X.shape)
# save GCN embeddings
save_embeddings(X, "./data/Ant/gcn_emb.emd")