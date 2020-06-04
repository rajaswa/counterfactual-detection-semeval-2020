# IMPORTS
from counterfactual_detection.data import *
from counterfactual_detection.models import *
from counterfactual_detection.utils import *
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

# SEED
seed_val = 1234
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# DEVICE
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("using ", device)

# CONFIG
model_name = "bert-base-uncased"
epochs = 10
learning_rate = 1e-3
bs = 32
fname = model_name + "_task_1_"
SAVE_PATH = "./"

# DATA LOADING
data = read_data("subtask-1/train.csv")
train_df, valid_df = train_test_split(data, random_state=seed_val, test_size=0.1)
train_df.reset_index(inplace=True)
valid_df.reset_index(inplace=True)

test_df = pd.read_csv("subtask-1/test.csv")
gold_label = [0] * len(test_df)
test_df["gold_label"] = gold_label

tokenizer = transformers.BertTokenizer.from_pretrained(model_name)

train_dset = dataset_task1(train_df, 512, tokenizer)
val_dset = dataset_task1(valid_df, 512, tokenizer)
test_dset = dataset_task1(test_df, 512, tokenizer)

train_loader = DataLoader(train_dset, batch_size=bs, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dset, batch_size=bs, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dset, batch_size=bs, shuffle=False, num_workers=0)

# MODEL
transformer = transformers.BertModel.from_pretrained(
    model_name, output_hidden_states=True, output_attentions=True
)
attention_cnn = AttentionCNN(num_heads=12 * 3)
ModelBase = ModelBase(
    transformer=transformer, cnn=attention_cnn, layer_list=[-1, -2, -3]
)
model = Task1Model(ModelBase)
model = model.to(device)

for name, param in model.ModelBase.transformer.named_parameters():
    if ("pooler" not in name) & ("layer.11" not in name):
        param.requires_grad = False

optimizer = transformers.AdamW(
    model.parameters(), lr=learning_rate, eps=10e-8, weight_decay=1e-2
)

criterion = nn.BCEWithLogitsLoss()

# TRAINING
print("\nTraining started ...\n")
train_loss = []
validation_loss = []
precision_list, f1_list, recall_list, accuracy_list = [], [], [], []
max_f1, max_prec, max_recall, max_accuracy = -999, -999, -999, -999
min_val_loss = 9999
start_time = time.time()

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        input_ids, attention_mask, labels = data
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.reshape(-1), labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 100 == 99:  # print every 25 mini-batches
            print(
                "[%d, %5d] loss: %.3f time: %.3f"
                % (epoch + 1, i + 1, running_loss / 100, time.time() - start_time)
            )
            running_loss = 0.0

    print("\nEPOCH ", epoch + 1, " TRAIN LOSS = ", epoch_loss / len(train_dset))
    train_loss.append(epoch_loss / len(train_dset))

    val_loss = 0.0
    model.eval()
    preds = []
    ground_truth = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input_ids, attention_mask, labels = data
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.reshape(-1), labels.float())
            val_loss += loss.item()

            sig = nn.Sigmoid()
            predicted = sig(logits.reshape(-1))
            for pred in predicted:
                preds.append(int(round(pred.item())))
            for label in labels:
                ground_truth.append(int(label.item()))

    print("EPOCH ", epoch + 1, " VAL LOSS = ", val_loss / len(val_dset))
    validation_loss.append(val_loss / len(val_dset))
    model.train()

    preds = np.array(preds)
    ground_truth = np.array(ground_truth)
    prec, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, preds, average="binary"
    )
    accuracy = accuracy_score(ground_truth, preds)
    precision_list.append(prec)
    f1_list.append(f1)
    recall_list.append(recall)
    accuracy_list.append(accuracy)
    print(
        "EPOCH ",
        epoch + 1,
        "VAL PREC:",
        prec,
        "REC:",
        recall,
        "F1:",
        f1,
        "ACC:",
        accuracy,
        "\n",
    )

    if max_f1 < f1:
        print("Model optimized, saving weights ...\n")
        torch.save(model.state_dict(), SAVE_PATH + fname + ".pt")
        torch.save(ModelBase.state_dict(), SAVE_PATH + fname + "_model_base.pt")
        min_val_loss = val_loss / len(val_dset)
        max_f1, max_prec, max_recall, max_accuracy = f1, prec, recall, accuracy


# PLOTS
fig = plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.legend()
plt.show()
fig.savefig(SAVE_PATH + fname + "loss.png", dpi=400)

fig = plt.figure()
plt.plot(precision_list, label="Precision")
plt.plot(recall_list, label="Recall")
plt.plot(f1_list, label="F1")
plt.plot(accuracy_list, label="Accuracy")
plt.legend()
plt.show()
fig.savefig(SAVE_PATH + fname + "metrics.png", dpi=400)


# PREDICTION
model.load_state_dict(torch.load(SAVE_PATH + fname + ".pt"))
model.eval()
preds = []
ground_truth = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        input_ids, attention_mask, labels = data
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        sig = nn.Sigmoid()
        predicted = sig(logits.reshape(-1))
        for pred in predicted:
            preds.append(int(round(pred.item())))
        for label in labels:
            ground_truth.append(int(label.item()))

del test_df["sentence"]
del test_df["gold_label"]
test_df["pred_label"] = preds
test_df.to_csv(SAVE_PATH + fname + "_test_post.csv", index=False)
