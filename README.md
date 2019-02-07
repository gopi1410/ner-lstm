### Steps to run (for English using coNLL dataset, tested on TF 1.4)

#### (Same model should work for Hindi provided you have the dataset and transliterate it to Latin script and then pass to the model)

1. Copy the glove.6B.50d.txt to embeddings folder (50 dimensional glove vectors. Can use other dimensions too)  
[Download link](http://nlp.stanford.edu/data/glove.6B.zip)

2. cd into embeddings folder and run:  
   **python glove_model.py --restore glove.6B.50d.txt --dimension 50 --corpus a --glove_path /**

(Here dimension should be the dimension of your word embedding file, golve.6B.50d.txt in this case. Also corpus and glove path arg params are required in the script, hence they can be initialized to any value. TODO : make as not required args)  
(If you're using word2vec model embeddings, use the wordvec_model.py file instead)

3. The above will generate a pkl file in the root directory (outside embeddings).   **glovevec_model_50.pkl** should be generated now.

4. Now we will generate the embeddings pickle file used by the model. For this we need to resize our data to a fixed max sentence length.  
The coNLL dataset is placed in data/eng folder. First we can find max sentence size in this dataset using  
   **python size_max.py -i eng.train.txt**

Next we need to resize the sentences to a fixed size using resuze_input.py (500 length sentences in this case i.e. 500 words max)  
   **python resize_input.py --input eng.train.txt --output eng.train.resized.txt --trim 500**  
   **python resize_input.py --input eng.testa.txt --output eng.testa.resized.txt --trim 500**  
   **python resize_input.py --input eng.testb.txt --output eng.testb.resized.txt --trim 500**  

5. Now we will use this resized data files to generate embeddings using the embeddings/get_conll_embeddings.py (get_icon_embeddings for icon format dataset). We use the earlier generated 50-0dimension embedding model file, glovevec_model_50.pkl, as an input here  
   **python get_conll_embeddings.py --train eng.train.resized.txt --test_a eng.testa.resized.txt --test_b eng.testb.resized.txt --sentence_length 500 --use_model glovevec_model_50.pkl --model_dim 50**

This should have generated an embed and a tag pickle (.pkl) files in embeddings folder for each of train, test_a and test_b (6 files in total)

6. Now we run our model. As per the paper, they add extra 11 dimensions to the embeddings. So the input dimension should be 11 + word_embedding dimension chosen. In our case, it'll be 11 + 50 (since 50-dim glove vectors were chosen) = 61.  
Sentence length should be the resized length that we set above or max sentence length in our dataset  
Class size should be the number of output tags (5 in case of coNLL - I-PER, I-LOC, I-ORG, I-MISC, O)  
   **python model.py --word_dim 61 --sentence_length 500 --class_size 5 --batch_size 256**
