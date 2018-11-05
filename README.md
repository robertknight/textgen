# textgen

textgen generates randomized text following the style and content of a training
text.

This project is an implementation of the 2014 paper ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850) using PyTorch.

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

## References

[Generating Sequences With Recurrent Neural Networks (arXiv preprint)](https://arxiv.org/abs/1308.0850)

[The Unreasonable Effectiveness of Recurrent Neural
Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
