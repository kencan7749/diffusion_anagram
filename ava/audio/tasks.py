"""The audio track's one task, registered where the search layer looks.

Time reversal is the audio counterpart of the flip: `ava.audio.engine` averages
the noise estimates of the forward branch and the time-flipped branch, exactly
Visual Anagrams' construction with a different involution. The two slots are
symmetric for the same reason a flip's are -- swapping the prompts gives the
same sound played the other way -- so both carry the role `subject` and share
one vocabulary.

`paper` is `audio`: the views are not in either paper's registry, and
`ava.image.views` must never be asked to build them. `ava.audio.perceive`
defines what a listener hears.
"""

from __future__ import annotations

from ava.image.tasks import AUDIO, SUBJECT, IllusionTask, Slot, register_task

TIME_REVERSE = IllusionTask(
    name="time_reverse",
    paper=AUDIO,
    slots=(Slot("forward", SUBJECT), Slot("reverse", SUBJECT)),
    view_names=("forward", "reverse"),  # keys of ava.audio.perceive.VIEWS
    reduction="mean",
    citation="ava/audio/engine.py: Visual Anagrams' sampler with time reversal "
    "as the view (Step 0b for the view, Step 1 for the first anagrams)",
)

# The frequency hybrid: Factorized Diffusion's hybrid image with a low-pass
# in place of the blur. `near` is what is heard in the room (the whole
# signal), `far` what reaches the next room (below 750 Hz). The sampler's
# split lives in `ava.audio.bands` and needs a fitted projector, which is why
# this task has a `--projector` argument and no default. Slots are not
# symmetric: swapping the prompts asks for a different sound.
FREQ_HYBRID_750 = IllusionTask(
    name="freq_hybrid_750",
    paper=AUDIO,
    slots=(Slot("near", SUBJECT), Slot("far", SUBJECT)),
    view_names=("near", "far_750"),
    reduction="sum",
    citation="ava/audio/bands.py: Factorized Diffusion's hybrid with a fitted "
    "low-pass on AudioLDM 2's latent (Step 0c for the view)",
)

# The time jigsaw: Visual Anagrams' jigsaw with time in place of the plane.
# `whole` is the recording as made; `shuffled` is the same recording cut into
# four equal blocks and spliced in the fixed order of ava.audio.permute
# (seed 0, a derangement). Not symmetric. The permutation is a constant, not
# drawn per run, so `randomised` stays False; runs still record it.
TIME_JIGSAW_4 = IllusionTask(
    name="time_jigsaw_4",
    paper=AUDIO,
    slots=(Slot("whole", SUBJECT), Slot("shuffled", SUBJECT)),
    view_names=("whole", "jigsaw_4"),
    reduction="mean",
    citation="ava/audio/engine.py BlockPermute: Visual Anagrams' sampler with a "
    "block permutation of time as the view (Step 0b --view jigsaw_4)",
)

AUDIO_TASKS: tuple[IllusionTask, ...] = (TIME_REVERSE, FREQ_HYBRID_750, TIME_JIGSAW_4)

for _task in AUDIO_TASKS:
    register_task(_task)
