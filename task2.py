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
fname = model_name + "_task_2_"
SAVE_PATH = "./"

# DATA LOADING
data = read_data("subtask-2/train.csv")
train_df, valid_df = train_test_split(data, random_state=seed_val, test_size=0.1)
train_df.reset_index(inplace=True)
valid_df.reset_index(inplace=True)

test_df = pd.read_csv(".subtask-2/test.csv")
test_label = [0] * len(test_df)
(
    test_df["antecedent_startid"],
    test_df["antecedent_endid"],
    test_df["consequent_startid"],
    test_df["consequent_endid"],
) = (test_label, test_label, test_label, test_label)

tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)

train_dset = dataset_task2(train_df, 512, tokenizer)
val_dset = dataset_task2(valid_df, 512, tokenizer)
test_dset = dataset_task2(test_df, 512, tokenizer)

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
ModelBase.load_state_dict(torch.load(SAVE_PATH + model_name + "_task_1__model_base.pt"))
model = Task2Model(ModelBase)
model = model.to(device)

for name, param in model.ModelBase.transformer.named_parameters():
    if ("pooler" not in name) & ("layer.11" not in name):
        param.requires_grad = False

optimizer = transformers.AdamW(
    model.parameters(), lr=learning_rate, eps=10e-8, weight_decay=1e-2
)

criterion = nn.SmoothL1Loss(reduction="mean")

# TRAINING
print("\nTraining started ...\n")

train_loss = []
validation_loss = []
min_val_loss = 99999
start_time = time.time()

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(train_loader):
        input_ids, attention_mask, labels, l = data
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels.float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 25 == 24:  # print every 25 mini-batches
            print(
                "[%d, %5d] loss: %.7f time: %.3f"
                % (epoch + 1, i + 1, running_loss / 25 / bs, time.time() - start_time)
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
            input_ids, attention_mask, labels, l = data
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels.float())
            val_loss += loss.item()

    print("EPOCH ", epoch + 1, " VAL LOSS = ", val_loss / len(val_dset))
    validation_loss.append(val_loss / len(val_dset))
    model.train()

    if val_loss < min_val_loss:
        print("Model optimized, saving weights ...\n")
        torch.save(model.state_dict(), "./" + fname + ".pt")
        min_val_loss = val_loss

# PLOTS
fig = plt.figure()
plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")
plt.legend()
plt.show()
fig.savefig("./" + fname + "loss.png", dpi=400)


# PREDICTION
model.load_state_dict(torch.load("./" + fname + ".pt"))
model.eval()
ant_start, ant_end, cons_start, cons_end = [], [], [], []

with torch.no_grad():
    for i, data in enumerate(test_loader):
        input_ids, attention_mask, labels, l = data
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        for i in range(len(logits)):
            logit = logits[i]
            length = l[i].item()
            ant_start.append(int(logit[0].item() * length))
            ant_end.append(int(logit[1].item() * length))
            cons_start.append(int(logit[2].item() * length))
            cons_end.append(int(logit[3].item() * length))

sub_df = pd.DataFrame(
    columns=[
        "sentenceID",
        "antecedent_startid",
        "antecedent_endid",
        "consequent_startid",
        "consequent_endid",
    ]
)
sub_df["sentenceID"] = test_df["sentenceID"]
sub_df["antecedent_startid"] = ant_start
sub_df["antecedent_endid"] = ant_end
sub_df["consequent_startid"] = cons_start
sub_df["consequent_endid"] = cons_end

sub_df.to_csv(SAVE_PATH + fname + "_test.csv", index=False)
