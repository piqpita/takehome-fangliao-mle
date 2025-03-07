from sentence_transformers import SentenceTransformer

class SentenceEmbeddingModel:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        return self.model.encode(sentences)

if __name__ == "__main__":
    model = SentenceEmbeddingModel()
    test_sentences = ["Hello world!", "Shop, snap and play to earn free gift cards with Fetch!"]
    embeddings = model.encode(test_sentences)
    print(embeddings)