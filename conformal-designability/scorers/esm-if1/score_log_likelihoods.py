import os
from pathlib import Path

import numpy as np
import torch
from biotite.sequence.io.fasta import FastaFile, get_sequences
from multichain_util import (
    extract_coords_from_complex,
    score_sequence_in_complex,
)
from tqdm import tqdm
from util import (
    extract_coords_from_structure,
    load_coords,
    load_structure,
    score_sequence,
)

"""
Base code for scoring sequences given a fixed backbone using ESM-IF1 model.
Mostly boilerplate code from the ESM-IF1 + varunshankar/structural-evolution repository.
TODO: Will need to be more customized for extracting custom scores from model to
calibrate CP on.
"""


def get_native_seq(pdbfile, chain):
    structure = load_structure(pdbfile, chain)
    _, native_seq = extract_coords_from_structure(structure)
    return native_seq


def score_singlechain_backbone(model, alphabet, args):
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    coords, native_seq = load_coords(args.pdbfile, args.chain)
    print("Native sequence loaded from structure file:")
    print(native_seq)
    print("\n")
    ll, _ = score_sequence(model, alphabet, coords, native_seq)
    print("Native sequence")
    print(f"Log likelihood: {ll:.2f}")
    print(f"Perplexity: {np.exp(-ll):.2f}")
    print("\nScoring variant sequences from sequence file..\n")
    infile = FastaFile()
    infile.read(args.seqpath)
    seqs = get_sequences(infile)
    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outpath, "w") as fout:
        fout.write("seqid,log_likelihood\n")
        for header, seq in tqdm(seqs.items()):
            ll, _ = score_sequence(model, alphabet, coords, str(seq))
            fout.write(header + "," + str(ll) + "\n")
    print(f"Results saved to {args.outpath}")


def score_multichain_backbone(model, alphabet, args):
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    structure = load_structure(args.pdbfile)
    coords, native_seqs = extract_coords_from_complex(structure)
    target_chain_id = args.chain
    native_seq = native_seqs[target_chain_id]
    order = args.order

    print("Native sequence loaded from structure file:")
    print(native_seq)
    print("\n")

    ll_complex, ll_targetchain = score_sequence_in_complex(
        model,
        alphabet,
        coords,
        native_seqs,
        target_chain_id,
        native_seq,
        order=order,
    )
    print("Native sequence")
    print(f"Log likelihood of complex: {ll_complex:.2f}")
    print(f"Log likelihood of target chain: {ll_targetchain:.2f}")
    print(f"Perplexity: {np.exp(ll_complex):.2f}")

    print("\nScoring variant sequences from sequence file..\n")
    infile = FastaFile()
    infile.read(args.seqpath)
    seqs = get_sequences(infile)
    Path(args.outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(args.outpath, "w") as fout:
        fout.write("seqid,log_likelihood, log_likelihood_target\n")
        for header, seq in tqdm(seqs.items()):
            ll_complex, ll_targetchain = score_sequence_in_complex(
                model,
                alphabet,
                coords,
                native_seqs,
                target_chain_id,
                str(seq),
                order=order,
            )
            fout.write(
                header + "," + str(ll_complex) + "," + str(ll_targetchain) + "\n"
            )
    print(f"Results saved to {args.outpath}")


def get_model_checkpoint_path(filename):
    # Expanding the user's home directory
    return os.path.expanduser(f"~/.cache/torch/hub/checkpoints/{filename}")
