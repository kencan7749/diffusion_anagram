"""A single point in the audio anagram's search space.

Kept separate from `ava.spec.CandidateSpec` rather than generalising it. The
image spec carries `prompt_low` / `prompt_high` and a blur sigma, which name
spatial frequencies; the audio one carries a duration and two prompts that name
directions of time. Forcing one dataclass to mean both would make every field
read as a euphemism, and the uid -- which is a hash of the fields -- would stop
being comparable across runs the moment the shared shape changed.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

# Stable Audio Open's transformer always denoises 1024 latent frames; the
# requested duration is realised by cropping the decoded waveform. 10 s keeps a
# clip inside CLAP's own analysis window, so the judge hears all of what is
# scored.
DURATION_S = 10.0
GUIDANCE_SCALE = 7.0
NUM_INFERENCE_STEPS = 100
NEGATIVE_PROMPT = "low quality"


@dataclass
class AudioCandidateSpec:
    """One candidate: what it should sound like forwards, and backwards."""

    prompt_forward: str
    prompt_reverse: str
    seed: int = 0
    duration_s: float = DURATION_S
    guidance_scale: float = GUIDANCE_SCALE
    num_inference_steps: int = NUM_INFERENCE_STEPS
    negative_prompt: str = NEGATIVE_PROMPT

    @property
    def prompts(self) -> list[str]:
        """Ordered to match the views (forward, reverse)."""
        return [self.prompt_forward, self.prompt_reverse]

    def uid(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
