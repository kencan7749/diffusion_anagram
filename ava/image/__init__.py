"""Hybrid-image track: one picture that reads as two, far and near.

The counterpart of `ava.audio`. Nothing is re-exported here on purpose --
`engine` pulls in DeepFloyd through diffusers and `judge` pulls in CLIP, and
importing `ava.image` should not pay for either. Import from the exact
submodule instead.

What is shared with the audio track lives one level up: `ava.spec`,
`ava.vocab`, `ava.propose` and `ava.metric` know nothing about pixels.
"""
