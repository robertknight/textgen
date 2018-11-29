"""
Generate random text "inspired" by a training text.

This is an implementation of the paper "Generating Sequences with Recurrent
Neural Networks" [1].

[1] https://arxiv.org/abs/1308.0850
"""

import argparse

from progress.bar import Bar
import torch
import torch.nn as nn
import numpy as np

from util import Vocabulary
from lstm import LSTM


def gen_training_sequences(text, seq_length, batch_size, vocab):
    """
    Generate training sequences, grouped into ``batch_size`` "threads".

    The input text is split into ``batch_size`` chunks, each of which is split
    into ``seq_length`` training sequences.

    Returns (seq, batch, char index) tensor.
    """
    seqs = []
    thread_length = len(text) // batch_size
    seqs_per_batch = thread_length // seq_length

    for thread in range(batch_size):
        thread_offset = thread * thread_length

        for seq_idx in range(seqs_per_batch):
            seq_offset = thread_offset + (seq_idx * seq_length)
            seq = text[seq_offset : seq_offset + seq_length + 1]
            seq = torch.tensor(vocab.encode_as_indexes(seq))
            seqs.append(seq)

    seqs = torch.stack(seqs)
    seqs = torch.reshape(seqs, (batch_size, seqs_per_batch, seq_length + 1))

    # Convert from (thread, seq, char) to (seq, thread, char).
    seqs = torch.transpose(seqs, 0, 1)

    assert seqs.shape == (seqs_per_batch, batch_size, seq_length + 1)

    return seqs


def train(model, train_seqs, epochs, vocab, device, checkpoint_path):
    seqs_per_batch, batch_size, seq_length = train_seqs.shape
    optimizer = torch.optim.Adam(model.parameters())
    steps_between_state_reset = 100
    zero_state = torch.zeros(
        (model.lstm_layers, batch_size, model.lstm_size),
        device=device,
        requires_grad=False,
    )
    min_loss = None

    def prepare_batch(seqs):
        # seqs: (batch, seq)
        y_true = seqs.to(device)
        # Convert from (batch, seq) to (seq, batch).
        y_true = torch.transpose(y_true, 0, 1)

        # Get inputs, which consist of training sequence minus the last char.
        x = seqs[:, :-1]
        # One-hot encode training sequences.
        x = vocab.eye[x]
        # Convert from (batch, seq, char) to (seq, batch, char)
        x = torch.transpose(x, 0, 1)
        x = torch.cat((torch.zeros(1, batch_size, vocab.size), x))
        x = x.to(device)

        assert y_true.shape == (seq_length, batch_size)
        assert x.shape == (seq_length, batch_size, vocab.size)

        return (x, y_true)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}. Min loss {min_loss}")

        steps_since_state_reset = 0

        hidden_state = zero_state.clone().detach()
        cell_state = zero_state.clone().detach()

        batch_progress = Bar(f"Training", max=seqs_per_batch, hide_cursor=False)
        for seq in range(seqs_per_batch):
            x, y_true = prepare_batch(train_seqs[seq])

            # model inputs: (seq, batch, features)
            # model outputs: (seq, batch, hidden_size)

            optimizer.zero_grad()
            y_pred, (hidden_state, cell_state) = model(x, (hidden_state, cell_state))
            y_pred = y_pred.reshape((seq_length * batch_size, -1))
            y_true = y_true.reshape((seq_length * batch_size))
            loss = torch.nn.functional.cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
            batch_progress.next()

            if min_loss is None or loss.item() < min_loss:
                min_loss = loss.item()

            # Retain the hidden and cell state between each batch so that the
            # model can learn long-range dependencies. The state is reset every
            # N steps to avoid retaining information about sequences very far
            # in the past.
            steps_since_state_reset += 1
            if steps_since_state_reset >= steps_between_state_reset:
                steps_since_state_reset = 0
                hidden_state = zero_state.clone().detach()
                cell_state = zero_state.clone().detach()
            else:
                hidden_state = hidden_state.clone().detach()
                cell_state = cell_state.clone().detach()
        batch_progress.finish()
        torch.save(model.state_dict(), checkpoint_path)

        print(f"Epoch {epoch} min loss {min_loss}")


def sample(model, seed_seq, sample_length, device, temperature, vocab):
    batch_size = 1

    # Add null row at start of input sequence to match training data.
    input_ = torch.cat((torch.zeros((1, vocab.size)), seed_seq))
    input_ = torch.unsqueeze(input_, batch_size).to(device)

    hidden_shape = (model.lstm_layers, batch_size, model.lstm_size)
    cell_state = torch.zeros(hidden_shape, device=device)
    hidden_state = torch.zeros(hidden_shape, device=device)

    generated_seq = ""

    while len(generated_seq) < sample_length:
        with torch.no_grad():
            output, (hidden_state, cell_state) = model(
                input_, (hidden_state, cell_state)
            )
            # Convert (seq, batch, class) to (batch, seq, class).
            output = torch.transpose(output, 0, 1)
            char_probs = torch.softmax(output[0, -1] / temperature, dim=0)
            output_char_idx = np.random.choice(vocab.size, p=char_probs.cpu().numpy())
            output_char = vocab.terms[output_char_idx]
            input_ = vocab.encode(output_char)
            # reshape from (seq_len, features) to (seq_len, batch, features)
            input_ = input_.reshape(1, 1, -1).to(device)
            generated_seq += output_char
    return vocab.decode(seed_seq) + generated_seq


class Model(nn.Module):
    def __init__(self, vocab_size, use_custom_lstm=False):
        super().__init__()
        self.lstm_size = 256

        if use_custom_lstm:
            print("Using custom LSTM module")
            lstm_class = LSTM
            self.lstm_layers = 1
        else:
            lstm_class = nn.LSTM
            self.lstm_layers = 3

        self.lstm = lstm_class(
            input_size=vocab_size,
            hidden_size=self.lstm_size,
            num_layers=self.lstm_layers,
        )
        self.linear = nn.Linear(in_features=self.lstm_size, out_features=vocab_size)

    def forward(self, x, *args):
        y, (hidden_state, cell_state) = self.lstm(x, *args)
        y = self.linear(y)
        return y, (hidden_state, cell_state)


def train_command(args, training_text, vocab, model, device):
    seq_length = args.seq_length
    batch_size = args.batch_size
    train_seqs = gen_training_sequences(
        training_text, seq_length=seq_length, batch_size=batch_size, vocab=vocab
    )
    seq_count = train_seqs.shape[0]
    print(
        f"Training with {seq_count * batch_size} sequences of {seq_length} chars"
        f" with {seq_count} sequences per thread"
    )
    train(
        model,
        train_seqs=train_seqs,
        epochs=args.epochs,
        vocab=vocab,
        device=device,
        checkpoint_path="model.checkpoint.pt",
    )


def generate_command(args, vocab, model, device):
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)
    while True:
        print("Enter starting text for generated output:")
        seed_str = input("> ")
        seed = vocab.encode(seed_str)
        generated_seq = sample(
            model,
            seed,
            sample_length=args.sample_length,
            device=device,
            temperature=args.temperature,
            vocab=vocab,
        )
        print(generated_seq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        choices=("cpu", "cuda"),
        help="Device to execute training on",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--custom-lstm",
        dest="use_custom_lstm",
        action="store_true",
        default=True,
        help="Use custom LSTM implementation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Define "train" command.
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("input", help="Input training text file")
    train_parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        help="Number of training epochs",
        default=30,
    )
    train_parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size (multiples of 32 recommended)",
        default=32,
    )
    train_parser.add_argument(
        "--seq-length",
        dest="seq_length",
        type=int,
        help="Training sequence length",
        default=90,
    )

    # Define "generate" command.
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("input", help="Input text file used at training time")
    generate_parser.add_argument(
        "--model",
        dest="model",
        type=str,
        help="Path of trained model to use",
        default="model.checkpoint.pt",
    )
    generate_parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        help="Softmax temperature (higher = more random outputs)",
        default=1.0,
    )
    generate_parser.add_argument(
        "--sample-length",
        dest="sample_length",
        type=int,
        help="Length of samples to generate",
        default=200,
    )
    args = parser.parse_args()

    # Process common arguments.
    device = torch.device(args.device)
    training_text = "".join(open(args.input))
    training_text_chars = sorted(list(set(training_text + "$")))
    vocab = Vocabulary(training_text_chars)

    # Prepare model.
    model = Model(vocab.size, use_custom_lstm=args.use_custom_lstm)
    model.to(device)

    if args.command == "generate":
        generate_command(args, vocab, model, device)
    elif args.command == "train":
        train_command(args, training_text, vocab, model, device)


if __name__ == "__main__":
    main()
