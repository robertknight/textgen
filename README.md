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

## Differences from the paper

- The paper does not describe how parallelism should be achieved in training.
  This implementation splits the input text into ``batch_size`` "threads"
  and then divides each thread into individual training sequences.
  This allows training to be parallelized while allowing the model to learn
  long-range dependencies:

  ```
  Input text: "one two foo bar"
  Batch size: 2
  Threads: ["one two", "foo bar"]
  Batches: [["one", "foo"], ["two", "bar"]]
  ```

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
