import gensim
import numpy as np
import cPickle as pickle
def main():
    model_path = './word2vec/gameAndyuliaokudata.model'
    dictionary_save_path = './word2vec/gameAndyuliaokudata_dictionary.pkl'
    reverse_dictionary_save_path = './word2vec/gameAndyuliaokudata_reverse_dictionary.pkl'
    embeddingMatrix_save_path = './word2vec/gameAndyuliaokudata_embedding.npy'
    model = gensim.models.Word2Vec.load(model_path)
    gameAndyuliaokudata_dictionary = {c: i for i, c in enumerate(model.wv.index2word)}
    gameAndyuliaokudata_reverse_dictionary = dict(zip(gameAndyuliaokudata_dictionary.values(), gameAndyuliaokudata_dictionary.keys()))
    gameAndyuliaokudata_embeddingMatrix = []
    for zi, i in gameAndyuliaokudata_dictionary.items():
        array = model.wv[zi]
        gameAndyuliaokudata_embeddingMatrix.append(array)
    gameAndyuliaokudata_embeddingMatrix = np.array(gameAndyuliaokudata_embeddingMatrix)
    pickle.dump(gameAndyuliaokudata_dictionary, open(dictionary_save_path, 'w'))
    pickle.dump(gameAndyuliaokudata_reverse_dictionary, open(reverse_dictionary_save_path, 'w'))
    np.save(open(embeddingMatrix_save_path, 'w'), gameAndyuliaokudata_embeddingMatrix)
    aaa = 1

def test():
    path = './word2vec/gameAndyuliaokudata_dictionary.pkl'
    aaa = pickle.load(open(path, 'r'))
    for i, c in aaa.items():
        print i, c
if __name__ == '__main__':
    # main()
    test()