# Training Considerations

### Freezing Layers:
1. **If the entire network is frozen:**
   - We keep the transformer model’s weights as well as both of the classification task heads weights fixed.
   - This is useful when inferencing new inputs, or we'd like to preserve the pre-trained knowledge exactly as it is. For example, I already trained both tasks on lasy year's data and if it well represents what's happening today, I could freeze the whole network.

2. **If only the transformer backbone is frozen:**
   - We keep the transformer model’s weights fixed and only fine-tune the task-specific heads.
   - This lets the classification heads specialize/generalize for the specific tasks while still using pre-trained embeddings, leveraging the sentence transformer which was trained on a large and diverse sentence pairs.

3. **If only one task head is frozen:**
   - We only keep one of the task classification heads weights fixed to make sure it doesn’t change while training the other head.
   - This could be helpful if one task is more difficult and needs more generalization while the other one can be kept stable.

### Transfer Learning:

1. **choice of pre-trained model:**
	- There are different choices of pre-trained sentence transformer. For example, all-MiniLM-L6-v2 could provide a lightweight and fast inference which is a good trade-off between accuracy and speed. all-mpnet-base-v2 has shown excellent accuracy on different NLP tasks. paraphrase-MiniLM-L6-v2 which is trained and as well better fit for paraphrase detection tasks.
	- Depending on the deployment/performance requirements and specific tasks, different pre-trained model could be chosen.
2. **The layers you would freeze/unfreeze:**
	- Say we use all-mpnet-base-v2 model since it provides great sentence-level embeddings with high accuracy
	- I might start with freezing the transformer model to preserve general semantic knowledge and train the task heads. Save the best model as the benchmark
	- Then I'd unfreeze the last few transformer layers to allow deeper task-specific adjustments to see if any improvements on test set

