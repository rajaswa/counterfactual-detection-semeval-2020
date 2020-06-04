# IMPORTS
import torch
from torch import nn

# MODELS
class AttentionCNN(nn.Module):
    """CNN Model to extract Attention Embedding from Multi-Head Self-Attention Weights
    
    Input:
        Attention-Weights {torch.tensor} -- Multi-Head Self-Attention Weights;  shape: [batch_size, num_heads, sequence_length, sequence_length]
    
    Output:
        Attention Embedding -- shape: [batch_size, 512]
    """

    def __init__(self, num_heads, p_drop=0.4):
        """
        Arguments:
            num_heads {int} -- number of self-attention heads for input attention-weights
        
        Keyword Arguments:
            p_drop {float} -- dropout probability (default: {0.4})
        """
        super(AttentionCNN, self).__init__()

        # Activations
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3)

        # Layer1
        self.conv1 = nn.Conv2d(
            num_heads, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        self.bn1 = nn.BatchNorm2d(
            8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        # Layer2
        self.conv2 = nn.Conv2d(
            8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
        )
        self.bn2 = nn.BatchNorm2d(
            16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        # Dense
        self.fc1 = nn.Linear(16 * 56 * 56, 512)

        print("Attention-CNN Intialized")

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.maxpool(self.relu(self.bn2(self.conv2(out))))

        out = out.reshape(-1, 16 * 56 * 56)
        out = self.dropout(self.relu(self.fc1(out)))

        return out


class ModelBase(nn.Module):
    """Base Architecture Model to extract Attention Embedding and Pooled Embedding from transformer outputs
    
    Input:
        input_ids {torch.tensor} -- input token IDs for transformer model;  shape:  [batch_size, sequence_length]
        attention_mask {torch.tensor} -- attention mask for transformer self-attention;  shape:  [batch_size, sequence_length]
    
    Output:
        feature_embedding -- combined feature embedding from transformer model outputs;  shape:  [batch_size, 128] 
    """

    def __init__(self, transformer, cnn, layer_list=[-1]):
        """
        Arguments:
            transformer {PyTorch Model} -- transformer model to be used for feature extraction
            cnn {Model of class AttentionCNN} -- CNN model to be used over multi-head self-attention weights
        
        Keyword Arguments:
            layer_list {list} -- output layers to be used (default: {[-1]})
        """
        super(ModelBase, self).__init__()
        self.layers = layer_list
        self.transformer = transformer
        self.cnn = cnn
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.layernorm = nn.LayerNorm(512 * 2)
        self.fc1 = nn.Linear(768 * len(layer_list), 512)
        self.fc2 = nn.Linear(512 * 2, 128)
        print("ModelBase Initialized")

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        layers_concatenated = torch.cat([outputs[2][i] for i in self.layers], dim=2)
        mean_pooled_out = torch.mean(layers_concatenated, dim=1)
        attention_matrix = torch.cat([outputs[3][i] for i in self.layers], dim=1)

        pooled_embed = self.dropout(self.relu(self.fc1(mean_pooled_out)))
        attention_embed = self.cnn(attention_matrix)

        out_embed = torch.cat([pooled_embed, attention_embed], dim=1)
        out_embed = self.layernorm(out_embed)
        out_embed = self.relu(self.fc2(out_embed))

        return out_embed


class Task1Model(nn.Module):
    """Model for Counterfactual Detection Binary-Classification
    
    Input:
        input_ids {torch.tensor} -- input token IDs for transformer model;  shape:  [batch_size, sequence_length]
        attention_mask {torch.tensor} -- attention mask for transformer self-attention;  shape:  [batch_size, sequence_length]
    
    Output:
        feature_embedding -- combined feature embedding from transformer model outputs;  shape:  [batch_size, 128] 
    """

    def __init__(self, ModelBase):
        """
        Arguments:
            ModelBase {Model of class ModelBase} -- Base Architecture model to be used for feature extraction
        """
        super(Task1Model, self).__init__()

        self.ModelBase = ModelBase

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(128, 1)

        print("Task-1 Model Initialized")

    def forward(self, input_ids, attention_mask):
        out = self.ModelBase(input_ids=input_ids, attention_mask=attention_mask)
        out = self.fc1(self.dropout(out))

        return out


class Task2Model(nn.Module):
    """Model for Antecedent-Consequent Detection Regression
    
    Input:
        input_ids {torch.tensor} -- input token IDs for transformer model;  shape:  [batch_size, sequence_length]
        attention_mask {torch.tensor} -- attention mask for transformer self-attention;  shape:  [batch_size, sequence_length]
    
    Output:
        feature_embedding -- combined feature embedding from transformer model outputs;  shape:  [batch_size, 128] 
    """

    def __init__(self, ModelBase):
        """
        Arguments:
            ModelBase {Model of class ModelBase} -- Base Architecture model to be used for feature extraction
        """
        super(Task2Model, self).__init__()

        self.ModelBase = ModelBase

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(128, 4)

        print("Task-2 Model Initialized")

    def forward(self, input_ids, attention_mask):
        out = self.ModelBase(input_ids=input_ids, attention_mask=attention_mask)
        out = self.relu(self.fc1(self.dropout(out)))

        return out
