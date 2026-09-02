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

AUDIO_TASKS: tuple[IllusionTask, ...] = (TIME_REVERSE,)

for _task in AUDIO_TASKS:
    register_task(_task)
