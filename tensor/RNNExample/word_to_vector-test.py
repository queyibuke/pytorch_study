from torchnlp.word_to_vector import GloVe
vectors = GloVe()

print(vectors['hello'])
print(vectors['hello'].shape)