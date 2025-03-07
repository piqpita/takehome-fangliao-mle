import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

#define model, could be imported from task2 as well
class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self, model_name="all-mpnet-base-v2", num_classes_task1=15, num_classes_task2=3):
        super(MultiTaskSentenceTransformer, self).__init__()
        self.backbone = SentenceTransformer(model_name)
        embedding_dim = self.backbone.get_sentence_embedding_dimension()
        
        # Task A: Sentence classification. class sentences into different groups based on a pre-defined list, say we have 15 classes (like Internet Issue, Cannot login, Payment Issue, Reward Points... etc.).
        self.classifier_task1 = nn.Linear(embedding_dim, num_classes_task1)

        # Task B: Sentiment analysis. class sentences into positive/neutral/negative sentiment classes.
        self.classifier_task2 = nn.Linear(embedding_dim, num_classes_task2)

    def forward(self, sentences):
        embeddings = self.backbone.encode(sentences, convert_to_tensor=True)
        task1_output = self.classifier_task_a(embeddings)
        task2_output = self.classifier_task_b(embeddings)
        return task_a_output, task_b_output

# Define Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Example Multi-Task Dataset, need to convert labels into class numbers first, more samples need to be added to ensure models are well trained
class MultiTaskDataset(Dataset):
    def __init__(self, task):
        self.task = task
        self.data = self._generate_data()
    
    def _generate_data(self):
        if self.task == 'task1':  # Sentence classification
            return [("Internet is down", 0), ("I can't log in", 1), ("Payment failed", 2)] #more samples need to be added
        elif self.task == 'task2':  # Sentiment analysis
            return [("The service is amazing", 0), ("It’s okay", 1), ("Worst experience ever", 2)] #more samples need to be added
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        return text, torch.tensor(label)
        
# Freeze all layers of the transformer except for the classifier layers
def freeze_transformer_layers(model, freeze=True):
    for name, param in model.backbone.named_parameters():
        if freeze:
            param.requires_grad = False  # Freeze layers
        else:
            param.requires_grad = True  # Unfreeze layers

# Data Splitting (80% Train, 10% Validation, 10% Test)
def split_data(dataset, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskSentenceTransformer().to(device)

# Split datasets
train_dataset_task1, val_dataset_task1, test_dataset_task1 = split_data(MultiTaskDataset('task1'))
train_dataset_task2, val_dataset_task2, test_dataset_task2 = split_data(MultiTaskDataset('task2'))

# Create Dataloaders
batch_size = 4
train_loader_task1 = DataLoader(train_dataset_task1, batch_size=batch_size, shuffle=True)
val_loader_task1 = DataLoader(val_dataset_task1, batch_size=batch_size, shuffle=False)
test_loader_task1 = DataLoader(test_dataset_task1, batch_size=batch_size, shuffle=False)

train_loader_task2 = DataLoader(train_dataset_task2, batch_size=batch_size, shuffle=True)
val_loader_task2 = DataLoader(val_dataset_task2, batch_size=batch_size, shuffle=False)
test_loader_task2 = DataLoader(test_dataset_task2, batch_size=batch_size, shuffle=False)

# Optimizer & Loss Functions
learning_rate = 2e-5
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn_task1 = nn.CrossEntropyLoss()
loss_fn_task2 = nn.CrossEntropyLoss()

# Training Loop with Validation and Early Stopping
epochs = 10
early_stopping = EarlyStopping(patience=5)
best_val_loss = np.inf
# Freeze the transformer layers initially
freeze_transformer_layers(model, freeze=True)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    
    # Train on Task 1
    for sentences, labels in train_loader_task1:
        labels = labels.to(device)
        optimizer.zero_grad()
        task1_output, _ = model(sentences)
        loss = loss_fn_task1(task1_output, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Train on Task 2
    for sentences, labels in train_loader_task2:
        labels = labels.to(device)
        optimizer.zero_grad()
        _, task2_output = model(sentences)
        loss = loss_fn_task2(task2_output, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    # Validation Step
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for sentences, labels in val_loader_task1:
            labels = labels.to(device)
            task1_output, _ = model(sentences)
            loss = loss_fn_task1(task1_output, labels)
            total_val_loss += loss.item()

        for sentences, labels in val_loader_task2:
            labels = labels.to(device)
            _, task2_output = model(sentences)
            loss = loss_fn_task2(task2_output, labels)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / (len(train_loader_task1) + len(train_loader_task2))
    avg_val_loss = total_val_loss / (len(val_loader_task1) + len(val_loader_task2))

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    if early_stopping(avg_val_loss):
        print("Early stopping triggered!")
        break

print("Training finished!")