"""
Test Transformers whether they understand the masks correctly.
"""

import torch as th

from nntrainer.models import TransformerEncoder, TransformerEncoderConfig


# dont change these or else the test will break
from nntrainer.models.transformer_legacy import TransformerDecoder


BATCH_SIZE = 3
QUERY_LEN = 8
KEY_LEN = 2 * QUERY_LEN

# this can be changed
HIDDEN_DIM = 32


@th.no_grad()
def test_transformers() -> None:
    """
    Sanity check for our COOT transformers.
    """
    # setup config
    cfg = TransformerEncoderConfig(
        {"hidden_dim": HIDDEN_DIM, "num_layers": 1, "dropout": 0.1, "num_heads": 2, "pointwise_ff_dim": 0,
         "activation": "gelu", "norm": "layernorm_coot"})

    # my encoder handles self and cross attention
    tenc = TransformerEncoder(cfg)
    tenc.eval()

    # create input
    query = th.randn((BATCH_SIZE, QUERY_LEN, HIDDEN_DIM))

    # create 3 different masks for the 3 inputs in the batch: mask nothing, mask everything except first, mask half
    mask = th.zeros((BATCH_SIZE, QUERY_LEN)).bool()
    mask[1, 1:] = True
    mask[2, QUERY_LEN // 2:] = True

    # do first forward pass of self-attention
    output = tenc(query, mask)
    assert output.shape == query.shape, f"query shape {query.shape}, output shape {output.shape}"

    # modify the query
    query_new = th.clone(query)
    # first batch item: all elements should change if the last input sequence element changes
    query_new[0, -1] += 10
    # second batch item: nothing should change, only the first input stays the same and it's the only unmasked one
    # so all attention matrices will only attend to this constant element
    query_new[1, 1:] += 10
    # third batch item: first half (unmasked) input stays the same, so the output for those should also stay the same
    # for second half the input changes, so their attention masks change and therefore the output.
    query_new[2, QUERY_LEN // 2:] += 10
    truth = th.Tensor([[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]]).bool()

    # # input modified output to the transformer and see if everything is fine
    mod_output = tenc(query_new, mask)
    is_different = ((output - mod_output) ** 2).mean(-1) > 1e-8
    assert th.all(is_different == truth), f"Transformer doesn't mask correctly! {is_different}"

    # for cross-attention, should be able to replace key and value with something random with same results
    # only difference: if we change only the last query item in batch sequence 0, only this output element will change
    # as the keys stay the same.
    tdec = TransformerDecoder(cfg)
    tdec.eval()

    truth_cross = truth.clone()
    truth_cross[0, :-1] = False
    key = th.randn((BATCH_SIZE, KEY_LEN, HIDDEN_DIM))
    cross_mask = th.ones((BATCH_SIZE, KEY_LEN)).bool()
    cross_mask[:, :QUERY_LEN] = mask
    output = tdec(query, key, cross_mask)
    mod_output = tdec(query_new, key, cross_mask)
    is_different = ((output - mod_output) ** 2).mean(-1) > 1e-8
    assert th.all(is_different == truth_cross), f"Transformer doesn't mask correctly:\n{is_different}"


if __name__ == "__main__":
    test_transformers()
