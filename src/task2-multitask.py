import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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

if __name__ == "__main__":
    model = MultiTaskSentenceTransformer()
    sample_sentences = ["I got internet Issue and cannot connect.", "I love Fetch Rewards!"]
    out1, out2 = model(sample_sentences)
    print("Task 1 Output (Sentence Classification):", out1)
    print("Task 2 Output (Sentiment Analysis):", out2)
    
    
"""
Describe  changes made to the architecture to support multi-task learning:
    The key change is to add two task-specific heads on top of the shared sentence transformer backbone to handle taks1 and task2 respectively.
    
    The shared sentence transformer: the sentence transformer model remains the pre-trained all-mpnet-base-v2 model as of now. This part of the architecture is shared across both tasks, 
        meaning that the transformer processes the input sentence in the same way, outputting the same enbeddings regardless of the task.
        
    Two task-specific heads:
        For task1, sentence classification, a fully connected layer is added that takes the output embeddings from the sentence transformer and then convert them to the number of classes defined for Task1. 
        For example, for a sentence classification like intent classification problem with 15 different classes (like Internet Issue, Cannot login, Payment Issue, Reward Points...), the output of task1 is a 15 dimension vector with 15 logits.

        For task2, sentiment classification, a fully connected layer is added that takes the output embeddings from the sentence transformer and then convert them to the number of classes defined for Task2.
        Normally there will be three classes for a sentiment classification problem: Positive/Neutral/Negative, the output of task2 is a 3 dimension vector with 3 logits corresponding to each sentiment class
        
    Last but not least, a forward pass is so that both task-specific heads receive the same embeddings outputs from the transformer backbone. And they process these embeddings differently based on their specific task.
        
"""
