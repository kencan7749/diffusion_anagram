"""Audio anagram track: one sound that reads as two, forward and reversed.

Kept as a subpackage so the audio work does not disturb the hybrid-image
pipeline in `ava/`. Nothing is re-exported here on purpose: the CLAP judge
pulls in torch and transformers, and importing `ava.audio` should not pay for
that. Import from the exact submodule instead.
"""
