#!/usr/bin/env python

def mdp_to_feature_belief(b, mdp_codes, features):
    fbelief = {f: 0 for f in features}
    for mdpc, bval in zip(mdp_codes, b):
        for isin, f in zip(mdpc, features):
            if isin == 'o':
                fbelief[f] += bval
    return fbelief