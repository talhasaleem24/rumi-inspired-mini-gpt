import unittest

import torch

from attention import CausalSelfAttention
from model import MiniGPT
from prepare_rumi import prepare_poetry
from tokenizer import build_vocabulary, decode, encode


class PoetryPreparationTests(unittest.TestCase):

    def test_preserves_stanzas_and_removes_print_artifacts(self):
        source = (
            "PROEM. The Reed-Flute.\n"
            "\n"
            "  From reed-flute[55] hear the song.                 5\n"
            "  A second line follows.\n"
            "\n"
            "  A new stanza begins. 10\n"
            "THE END.\n"
        )

        result = prepare_poetry(source)

        self.assertEqual(
            result,
            "PROEM. The Reed-Flute.\n\n"
            "From reed-flute hear the song.\n"
            "A second line follows.\n\n"
            "A new stanza begins."
        )


class TokenizerTests(unittest.TestCase):

    def test_round_trip_and_unknown_token_handling(self):
        text = "Love\nis kind."
        stoi = build_vocabulary(text, min_frequency=1)

        self.assertEqual(decode(encode(text, stoi), stoi), text)

        limited_stoi = build_vocabulary("love love ", min_frequency=2)
        unknown_id = limited_stoi["<unk>"]
        encoded = encode("love ☃", limited_stoi)
        self.assertIn(unknown_id, encoded)
        self.assertIn("[UNK]", decode(encoded, limited_stoi))


class TransformerTests(unittest.TestCase):

    def test_attention_is_causal_and_normalized(self):
        attention = CausalSelfAttention(embed_dim=8, num_heads=2)
        _, weights = attention(torch.randn(2, 5, 8))

        self.assertTrue(torch.allclose(
            weights.sum(dim=-1),
            torch.ones_like(weights.sum(dim=-1)),
            atol=1e-6
        ))
        self.assertTrue(torch.equal(
            weights.masked_select(torch.triu(
                torch.ones(5, 5, dtype=torch.bool),
                diagonal=1
            ).view(1, 1, 5, 5).expand_as(weights)),
            torch.zeros_like(weights.masked_select(torch.triu(
                torch.ones(5, 5, dtype=torch.bool),
                diagonal=1
            ).view(1, 1, 5, 5).expand_as(weights)))
        ))

    def test_custom_vocabulary_forward_and_generation(self):
        model = MiniGPT(
            vocab_size=11,
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0
        )
        inputs = torch.randint(0, 11, (2, 6))
        logits, loss = model(inputs, inputs)

        self.assertEqual(logits.shape, (2, 6, 11))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

        generated = model.generate(
            inputs[:1, :2],
            max_new_tokens=3,
            temperature=1.0,
            top_k=5,
            forbidden_token_ids=[10]
        )
        self.assertEqual(generated.shape, (1, 5))
        self.assertNotIn(10, generated[0, 2:].tolist())

    def test_generation_rejects_invalid_sampling_options(self):
        model = MiniGPT(
            vocab_size=5,
            embed_dim=8,
            num_heads=2,
            num_layers=1
        )

        with self.assertRaises(ValueError):
            model.generate(torch.ones((1, 1), dtype=torch.long), 1, top_k=0)


if __name__ == "__main__":
    unittest.main()
