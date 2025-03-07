from sentence_transformers import SentenceTransformer

class SentenceEmbeddingModel:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        """
        Encodes input sentences into fixed-length embeddings.

        """
        return self.model.encode(sentences)

if __name__ == "__main__":
    model = SentenceEmbeddingModel()
    test_sentences = ["Hello world!", "Shop, snap and play to earn free gift cards with Fetch!"]
    embeddings = model.encode(test_sentences)
    print(embeddings)
    
"""

Describe choices I had to make:
    1. Choice of sentence transformer model:
        I chose all-mpnet-base-v2 because it ranks high in terms of accuracy;
        Although it's slower than those MiniLM models, it provides more richness with output enbeddings at length of 768 and can intake more length of texts.        
        
"""