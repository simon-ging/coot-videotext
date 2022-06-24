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
        if "vid_emb" not in h5:
            # backwards compatibility
            (f_vid_emb, f_vid_emb_before_norm, f_clip_emb, f_clip_emb_before_norm, f_vid_context,
             f_vid_context_before_norm, f_par_emb, f_par_emb_before_norm, f_sent_emb, f_sent_emb_before_norm,
             f_par_context, f_par_context_before_norm) = (
                "vid_norm", "vid", "clip_norm", "clip", "vid_ctx_norm", "vid_ctx",
                "par_norm", "par", "sent_norm", "sent", "par_ctx_norm", "par_ctx")
            data_collector = dict((key_target, np.array(h5[key_source])) for key_target, key_source in zip(
                ["vid_emb", "par_emb", "clip_emb", "sent_emb"], [f_vid_emb, f_par_emb, f_clip_emb, f_sent_emb]))
        else:
            # new version
            data_collector = dict((key, np.array(h5[key])) for key in ["vid_emb", "par_emb", "clip_emb", "sent_emb"])

    # compute retrieval
    print(retrieval.VALHEADER)
    retrieval.compute_retrieval(data_collector, "vid_emb", "par_emb")
    retrieval.compute_retrieval(data_collector, "clip_emb", "sent_emb")


if __name__ == "__main__":
    main()
