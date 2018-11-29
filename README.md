# textgen

textgen generates randomized text following the style and content of a training
text.

This project is an implementation of the first part of the 2014 paper ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850), which concerns
generating text sequences, using PyTorch.

Requires Python >= 3.6 and PyTorch 1.0. For training on large datasets, an
NVIDIA GPU is very helpful.

## Usage

```
pip install -r requirements.txt

# Train the model on a source text. The generated model is saved to
# "model.checkpoint.pt" at the end of each epoch.
python textgen train <source text>

# Run a REPL loop which reads input sequences and suggests completions.
python textgen generate <source text>
```

For best results, `<source text>` needs to be quite large (1MB+) and you will
need either a CUDA-supporting GPU or a lot of patience for training.

## Training tips

- Use the `--epochs` parameter to control the trade-off between amount of
  training time and accuracy of the learned model.
  On English text, the model typically learns to generate words and sentences
  after a few epochs over a ~1MB source text. In order to generate
  sensible-looking phrases 30+ epochs are typically needed.
- When training on a GPU, increasing the batch size via `--batch-size` will
  reduce the time per epoch, but tends to also reduce the learning rate.

  A batch size of 64 worked well in my tests on a single GPU.
- On English text, models with a loss of >2 produce samples with very little
  structure. When the loss is between 1 and 2 some basic word and sentence
  structure starts to emerge. Plausible phrases start to emerge when the loss
  gets below 0.5.

## Generation tips

- The model is saved after every epoch, so you can run the `generate` command
  while the model is still training.
- The `--temperature` option controls the randomness of the generated text.
  Values < 1.0 are more conservative, values > 1.0 allow the model to be more
  "creative" (output less-probable sequences according to the learned model).

## Differences from the paper

- The paper does not describe how parallelism should be achieved in training.
  This implementation splits the input text into ``batch_size`` "threads"
  and then divides each thread into individual training sequences.
  This allows training to be parallelized while allowing the model to learn
  long-range dependencies:

  Input text: `"one two foo bar"`
  Batch size: 2
  Threads: `["one two", "foo bar"]`
  Batches: `[["one", "foo"], ["two", "bar"]]`

- For multi-layered models, the model in the paper adds skip connections from
  the input to each hidden layer and from each hidden layer to the output.

  This implementation just uses the multi-layer support in PyTorch's LSTM where
  the input is connected to the first layer and each hidden layer is connected
  to the next.

## References

[Generating Sequences With Recurrent Neural Networks (arXiv preprint)](https://arxiv.org/abs/1308.0850)

[The Unreasonable Effectiveness of Recurrent Neural
Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
