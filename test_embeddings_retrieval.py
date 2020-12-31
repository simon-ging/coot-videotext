"""
Compute retrieval on provided coot embeddings.
"""

from pathlib import Path

import h5py
import numpy as np

from nntrainer import retrieval, utils


def main():
    parser = utils.ArgParser(description=__doc__)
    parser.add_argument("path_to_embeddings", type=str, help="Provide path to h5 embeddings file.")
    args = parser.parse_args()
    path_to_embeddings = Path(args.path_to_embeddings)
    print(f"Testing retrieval on embeddings: {path_to_embeddings}")

    # load embeddings
    with h5py.File(path_to_embeddings, "r") as h5:
        data_collector = dict((key, np.array(h5[key])) for key in ["vid_emb", "par_emb", "clip_emb", "sent_emb"])

    # compute retrieval
    print(retrieval.VALHEADER)
    retrieval.compute_retrieval(data_collector, "vid_emb", "par_emb")
    retrieval.compute_retrieval(data_collector, "clip_emb", "sent_emb")


if __name__ == "__main__":
    main()
