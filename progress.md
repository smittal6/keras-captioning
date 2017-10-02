# Automatic Image Captioning

- Currently, this repo holds the Keras implementation for Show and Tell. We plan to
  add Show, Attend and Tell soon.

## Here is the workflow:
### CNN Embedding{embed_image.py}
- {VGG16, ResNet50, InceptionV3} can be used for getting image embeddings.

### Word Embedding
- We need to obtain embeddding of words. 
  One-hot? Word2Vec? Other Embedding?
  Currently, not using any pretrained embedding. Let the model learn it.
  Could take a lot of time

### Language Model{model.py}
- Relating image embedding and language model. Currently, concatenating the
  vector embeddings, and let the LSTM output the final representation.

## Possible Experiments:

- To add the image embedding or Concatenate it
- LSTM/GRU units in the language model(not the embedding)
- Using pretrained embeddings like Glove(which the author reported as best),
  word2vec etc.
- Work on the final output of the LSTM or the sequence returned by setting
  return_sequence=True. For this, we would have to make changes in the data
  processing step.

## To do:
- Implement Beam Search
- Adding data generator
- Evaluation Step
- Mapping Captions to Vectors

## References:
https://keras.io/applications/
