# Prompts quoted from the papers, by task

Generated from `ava/image/paper_examples.py`; do not edit by hand. Each line is
a published figure's prompts, reassembled verbatim from the seed vocabulary's
style and subjects, in the task's slot order. VA = Geng et al. 2024 (Visual
Anagrams, arXiv 2311.17919v2); FD = Geng et al. 2025 (Factorized Diffusion,
arXiv 2404.11615v2). These seed the vocabulary as `source='paper'` arms with a
uniform prior; the figure is the arm's citation.

## `flip` — slots: identity / flip

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1 | `a painting of a red panda` / `a painting of kitchenware` |
| VA Fig. 1 | `a painting of a sloth` / `a painting of vases` |
| VA Fig. 1 | `a painting of a turtle` / `a painting of wine and cheese` |
| VA Fig. 1 | `a lithograph of a duck` / `a lithograph of a fish` |
| VA Fig. 1, 2, 8 | `an oil painting of people at a campfire` / `an oil painting of an old man` |
| VA Fig. 1 | `an oil painting of a bird` / `an oil painting of a ship` |
| VA Fig. 1 | `a drawing of a penguin` / `a drawing of a giraffe` |
| VA Fig. 1 | `a photo of an old woman` / `a photo of a young lady` |
| VA Fig. 5 | `a painting of a truck` / `a painting of a deer` |
| VA Fig. 5 | `a painting of a horse` / `a painting of a bird` |
| VA Fig. 5 | `a painting of a dog` / `a painting of an airplane` |
| VA Fig. 5 | `an ink drawing of a house` / `an ink drawing of a castle` |
| VA Fig. 5 | `a watercolor painting of an owl` / `a watercolor painting of a dog` |
| VA Fig. 5 | `a street art of a rabbit` / `a street art of a violin` |
| VA Fig. 10 | `a painting of a dog` / `a painting of a cat` |
| VA Fig. 10 | `a painting of a frog` / `a painting of a deer` |
| VA Fig. 10 | `a painting of a truck` / `a painting of a ship` |
| VA Fig. 10 | `a pop art of a giraffe` / `a pop art of a bird` |
| VA Fig. 10 | `a sketch of a cup` / `a sketch of a frog` |
| VA Fig. 10 | `an oil painting of a cat` / `an oil painting of a mouse` |
| VA Fig. 12 | `an oil painting of a museum` / `an oil painting of a quokka` |
| VA Fig. 12 | `an oil painting of a kitchen` / `an oil painting of a quokka` |
| VA Fig. 16 | `a photo of a wedding dress` / `a photo of an old woman` |
| VA Fig. 16 | `an oil painting of albert einstein` / `an oil painting of elvis` |
| VA Fig. 16 | `an oil painting of a red panda` / `an oil painting of a teddy bear` |
| VA Fig. 16 | `an oil painting of a kitchen` / `an oil painting of a botanical garden` |
| VA Fig. 16 | `a painting of a museum` / `a painting of a camel` |
| VA Fig. 16 | `a painting of a sculpture garden` / `a painting of a deer` |
| VA Fig. 16 | `a painting of a still life of trophies and awards` / `a painting of a quokka` |
| VA Fig. 16 | `a painting of wine and cheese` / `a painting of a pig` |
| VA Fig. 13 | `a painting of sunflowers` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a vampire` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of waterfalls` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a red panda` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a quokka` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a deer` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of elvis presley` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a garden` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a horse` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of marilyn monroe` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of a penguin` / `a painting of albert einstein` |
| VA Fig. 13 | `a painting of students in a classroom` / `a painting of albert einstein` |
| VA Fig. 14 | `a painting of sunflowers` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of a turkey` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of waterfalls` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of a red panda` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of a horse` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of a deer` / `a painting of abraham lincoln` |
| VA Fig. 14 | `a painting of elvis presley` / `a painting of michael jackson` |
| VA Fig. 14 | `a painting of albert einstein` / `a painting of michael jackson` |
| VA Fig. 14 | `a painting of a vampire` / `a painting of michael jackson` |
| VA Fig. 14 | `a painting of a quokka` / `a painting of michael jackson` |
| VA Fig. 14 | `a painting of jeans` / `a painting of michael jackson` |
| VA Fig. 14 | `a painting of a garden` / `a painting of michael jackson` |
| VA Fig. 15 | `a painting of a ship` / `a painting of a car` |
| VA Fig. 15 | `a painting of a horse` / `a painting of a cat` |
| VA Fig. 15 | `a painting of a ship` / `a painting of an airplane` |
| VA Fig. 15 | `a painting of a bird` / `a painting of a car` |

## `jigsaw` — slots: identity / jigsaw

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1, 8 | `an oil painting of a fruit bowl` / `an oil painting of a monkey` |
| VA Fig. 1 | `a watercolor of a kitten` / `a watercolor of a puppy` |
| VA Fig. 1, 8 | `a painting of houseplants` / `a painting of marilyn monroe` |
| VA Fig. 8 | `a painting of wine and cheese` / `a painting of a turtle` |
| VA Fig. 12 | `an oil painting of a woman staring out a window` / `an oil painting of abraham lincoln` |
| VA Fig. 12 | `an oil painting of a fruit bowl` / `an oil painting of albert einstein` |
| VA Fig. 12 | `an oil painting of flower arrangements` / `an oil painting of a quokka` |
| VA Fig. 17 | `a painting of a kitchen` / `a painting of a camel` |
| VA Fig. 17 | `a painting of a dining table` / `a painting of a polar bear` |
| VA Fig. 17 | `a painting of houseplants` / `a painting of an octopus` |
| VA Fig. 17 | `a painting of flower arrangements` / `a painting of a sloth` |

## `inner_circle` — slots: identity / inner_circle

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1 | `a pop art of albert einstein` / `a pop art of marilyn monroe` |
| VA Fig. 17 | `an oil painting of an old man` / `an oil painting of a chair` |

## `skew` — slots: identity / skew

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1, 8 | `an oil painting of a tudor portrait` / `an oil painting of a skull` |
| VA Fig. 1 | `an oil painting of a soldier` / `an oil painting of a houseplant` |

## `negate` — slots: identity / negate

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1 | `a lithograph of a teddy bear` / `a lithograph of a rabbit` |
| VA Fig. 1 | `a lithograph of a landscape` / `a lithograph of houseplants` |
| VA Fig. 1 | `a photo of a man` / `a photo of a woman` |
| VA Fig. 12 | `a blockprint of a red panda` / `a blockprint of elvis presley` |
| VA Fig. 12, 16 | `a lithograph of waterfalls` / `a lithograph of a table` |
| VA Fig. 16 | `an ink drawing of a red panda` / `an ink drawing of elvis` |
| VA Fig. 16 | `an ink drawing of waterfalls` / `an ink drawing of wine and cheese` |

## `three_view` — slots: identity / rotate_cw / rotate_ccw

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1 | `a painting of a teddy bear` / `a painting of a rabbit` / `a painting of waterfalls` |
| VA Fig. 1 | `a painting of einstein` / `a painting of elvis` / `a painting of houseplants` |
| VA Fig. 17 | `a painting of ancient ruins` / `a painting of a red panda` / `a painting of a teddy bear` |
| VA Fig. 17 | `a painting of a coral reef` / `a painting of a quokka` / `a painting of elvis` |
| VA Fig. 17 | `a painting of waterfalls` / `a painting of a red panda` / `a painting of elvis` |

## `four_view` — slots: identity / rotate_cw / rotate_180 / rotate_ccw

| citation | prompts (slot order) |
|---|---|
| VA Fig. 1, 11 | `an oil painting of a rabbit` / `an oil painting of a giraffe` / `an oil painting of a teddy bear` / `an oil painting of a bird` |

## `patch_permute` — slots: identity / patch_permute

| citation | prompts (slot order) |
|---|---|
| VA Fig. 6 | `a pencil sketch of a lemur` / `a pencil sketch of a kangaroo` |
| VA Fig. 6 | `a watercolor of a rabbit` / `a watercolor of a duck` |
| VA Fig. 17 | `a pencil sketch of a zebra` / `a pencil sketch of a motorcycle` |
| VA Fig. 6, 17 | `an oil painting of a young man` / `an oil painting of an old man` |

## `pixel_permute` — slots: identity / pixel_permute

| citation | prompts (slot order) |
|---|---|
| VA Fig. 6 | `a mosaic of a duck` / `a mosaic of a rabbit` |
| VA Fig. 17 | `a photo of a soccer ball` / `a photo of a panda` |

## `rotate_cw` — slots: identity / rotate_cw

| citation | prompts (slot order) |
|---|---|
| VA Fig. 8 | `an oil painting of a snowy mountain village` / `an oil painting of a horse` |
| VA Fig. 12 | `a watercolor painting of a village in the mountains` / `a watercolor painting of a ship` |
| VA Fig. 12 | `an oil painting of a theater` / `an oil painting of a library` |
| VA Fig. 16 | `a lithograph of a village in the mountains` / `a lithograph of a ship` |
| VA Fig. 16 | `a lithograph of a theater` / `a lithograph of a ship` |

## `hybrid` — slots: low / high

| citation | prompts (slot order) |
|---|---|
| FD Fig. 1 | `a photo of marilyn monroe` / `a photo of houseplants` |
| FD Fig. 1 | `a photo of an old man` / `a photo of a rabbit` |
| FD Fig. 1 | `a photo of a john lennon` / `a photo of new york city` |
| FD Fig. 1 | `a photo of a yin yang` / `a photo of rome` |
| FD Fig. 1 | `a photo of a teddy bear` / `a photo of mountains` |
| FD Fig. 1 | `a lithograph of a pig` / `a lithograph of waterfalls` |
| FD Fig. 1 | `a lithograph of a panda` / `a lithograph of flower arrangements` |
| FD Fig. 1 | `a lithograph of a deer` / `a lithograph of a ski slope in the alps` |
| FD Fig. 9 | `a photo of an old man` / `a photo of a rabbit` |
| FD Fig. 9 | `a watercolor of a panda` / `a watercolor of mountains` |
| FD Fig. 3 | `a headshot of albert einstein` / `a headshot of marilyn monroe` |
| FD Fig. 3 | `a photo of a snowy mountain village` / `a photo of a skull` |
| FD Fig. 16 | `oil painting style, a bumblebee` / `oil painting style, a bazaar` |
| FD Fig. 16 | `oil painting style, abraham lincoln` / `oil painting style, a flower arrangement` |
| FD Fig. 16 | `oil painting style, a bird` / `oil painting style, texture of feathers` |
| FD Fig. 16 | `oil painting style, a panda` / `oil painting style, the grand canyon` |
| FD Fig. 16 | `oil painting style, john lennon` / `oil painting style, the grand canyon` |
| FD Fig. 16 | `oil painting style, a panda` / `oil painting style, mountains` |
| FD Fig. 16 | `oil painting style, a panda` / `oil painting style, new york city` |
| FD Fig. 16 | `oil painting style, an old man` / `oil painting style, a bazaar` |
| FD Fig. 16 | `a photo of an old woman` / `a photo of houseplants` |
| FD Fig. 16 | `a photo of audrey hepburn` / `a photo of an english breakfast` |
| FD Fig. 16 | `a photo of an old woman` / `a photo of a library` |
| FD Fig. 16 | `a photo of gandhi` / `a photo of a forest` |
| FD Fig. 16 | `a photo of an old man` / `a photo of texture of granite` |
| FD Fig. 16 | `a photo of a panda` / `a photo of a barn` |
| FD Fig. 16 | `a photo of abraham lincoln` / `a photo of a bazaar` |
| FD Fig. 16 | `a photo of john lennon` / `a photo of a flower arrangement` |
| FD Fig. 16 | `a photo of gandhi` / `a photo of a sunset` |
| FD Fig. 16 | `a photo of an old man` / `a photo of houseplants` |
| FD Fig. 16 | `a photo of elvis` / `a photo of the grand canyon` |
| FD Fig. 16 | `a photo of an old woman` / `a photo of flower arrangements` |
| FD Fig. 16 | `a photo of a teddy bear` / `a photo of the grand canyon` |
| FD Fig. 16 | `a photo of abraham lincoln` / `a photo of a flower arrangement` |
| FD Fig. 16 | `a lithograph of a skull` / `a lithograph of waterfalls` |
| FD Fig. 16 | `a lithograph of a quokka` / `a lithograph of houseplants` |
| FD Fig. 16 | `a lithograph of houseplants` / `a lithograph of waterfalls` |
| FD Fig. 16 | `a lithograph of a skull` / `a lithograph of houseplants` |
| FD Fig. 16 | `a watercolor of king tut` / `a watercolor of a sunset` |
| FD Fig. 16 | `a watercolor of a panda` / `a watercolor of a library` |
| FD Fig. 16 | `a watercolor of a teddy bear` / `a watercolor of new york city` |
| FD Fig. 16 | `a watercolor of a bird` / `a watercolor of a bazaar` |
| FD Fig. 18 | `a photo of a skull` / `a photo of waterfalls` |
| FD Fig. 18 | `a photo of a teddy bear` / `a photo of a sunset` |
| FD Fig. 18 | `lithograph style, a bumblebee` / `lithograph style, a flower arrangement` |

## `triple_hybrid` — slots: low / mid / high

| citation | prompts (slot order) |
|---|---|
| FD Fig. 1, 2 | `a photo of a yin yang` / `a photo of a skull` / `a photo of waterfalls` |
| FD Fig. 1, 14 | `a photo of a pyramid` / `a photo of a dog` / `a photo of flower arrangements` |
| FD Fig. 1, 14 | `a photo of a yin yang` / `a photo of a rabbit` / `a photo of flower arrangements` |
| FD Fig. 1, 14 | `a photo of the eiffel tower` / `a photo of a quokka` / `a photo of houseplants` |
| FD Fig. 1, 14 | `a photo of a diamond` / `a photo of an old man` / `a photo of a fish` |
| FD Fig. 1, 14 | `a photo of a pyramid` / `a photo of a rabbit` / `a photo of flower arrangements` |

## `color_hybrid` — slots: gray / color

| citation | prompts (slot order) |
|---|---|
| FD Fig. 1 | `a watercolor of houseplants` / `a watercolor of the statue of liberty` |
| FD Fig. 1 | `a painting of a barn` / `a painting of a bumblebee` |
| FD Fig. 1 | `a painting of a flower arrangement` / `a painting of a teddy bear` |
| FD Fig. 5 | `a painting of a landscape` / `a painting of a tiger` |
| FD Fig. 5 | `a painting of a volcano` / `a painting of a rabbit` |
| FD Fig. 5 | `a painting of a dining table` / `a painting of a polar bear` |
| FD Fig. 5 | `oil painting style, the grand canyon` / `oil painting style, a bird` |
| FD Fig. 5 | `a photo of a bird` / `a photo of a frog` |
| FD Fig. 5 | `a photo of the grand canyon` / `a photo of a heart` |
| FD Fig. 17 | `a photo of a car` / `a photo of a ship` |
| FD Fig. 17 | `a photo of a ship` / `a photo of a frog` |
| FD Fig. 17 | `a watercolor of ancient ruins` / `a watercolor of an old man` |
| FD Fig. 17 | `a watercolor of flower arrangements` / `a watercolor of an old woman` |
| FD Fig. 17 | `a watercolor of a desert` / `a watercolor of a camel` |
| FD Fig. 17 | `a watercolor of flower arrangements` / `a watercolor of a duck` |
| FD Fig. 17 | `a watercolor of a landscape` / `a watercolor of a pig` |
| FD Fig. 17 | `a watercolor of a rainforest` / `a watercolor of a crocodile` |
| FD Fig. 17 | `a watercolor of waterfalls` / `a watercolor of a skull` |
| FD Fig. 17 | `a painting of a theater` / `a painting of a duck` |
| FD Fig. 17 | `a painting of city skyscrapers` / `a painting of a rabbit` |
| FD Fig. 17 | `oil painting style, mountains` / `oil painting style, a rabbit` |
| FD Fig. 17 | `oil painting style, the grand canyon` / `oil painting style, john lennon` |
| FD Fig. 17 | `oil painting style, a flower arrangement` / `oil painting style, a bird` |
| FD Fig. 17 | `oil painting style, a volcano` / `oil painting style, a duck` |
| FD Fig. 17 | `oil painting style, mountains` / `oil painting style, a tiger` |
| FD Fig. 17 | `an oil painting of an old man` / `an oil painting of students in a classroom` |
| FD Fig. 17 | `a lithograph of the grand canyon` / `a lithograph of a bumblebee` |
| FD Fig. 9 | `an oil painting of waterfalls` / `an oil painting of a tiger` |
| FD Fig. 9 | `a watercolor painting of flower arrangements` / `a watercolor painting of a duck` |
| FD Fig. 18 | `oil painting style, a library` / `oil painting style, a bumblebee` |
| FD Fig. 18 | `a watercolor of a flower arrangement` / `a watercolor of a bird` |
| FD Fig. 18 | `an oil painting of vases` / `an oil painting of a duck` |
| FD Fig. 18 | `an oil painting of a volcano` / `an oil painting of a rabbit` |

## `motion_hybrid` — slots: moving / still

| citation | prompts (slot order) |
|---|---|
| FD Fig. 1 | `a photo of a car` / `a photo of a canyon` |
| FD Fig. 1 | `a photo of a horse` / `a photo of a sports stadium` |
| FD Fig. 1 | `a painting of a dog` / `a painting of a beehive` |
| FD Fig. 6 | `a photo of sunflowers` / `a photo of a van gogh portrait` |
| FD Fig. 6 | `a photo of a rabbit` / `a photo of a duck` |
| FD Fig. 6 | `a photo of abraham lincoln` / `a photo of a mountain` |
| FD Fig. 6 | `a photo of a panda` / `a photo of a canyon` |
| FD Fig. 6 | `a photo of a car` / `a photo of a stadium` |
| FD Fig. 6 | `a photo of a teddy bear` / `a photo of new york city` |
| FD Fig. 15 | `a watercolor of a dog` / `a watercolor of a bazaar` |
| FD Fig. 15 | `oil painting style, a heart` / `oil painting style, the grand canyon` |
| FD Fig. 15 | `oil painting style, an old man` / `oil painting style, a forest` |
| FD Fig. 15 | `a photo of a bird` / `a photo of waterfalls` |
| FD Fig. 15 | `a photo of a teddy bear` / `a photo of ancient ruins` |
| FD Fig. 15 | `a photo of a skull` / `a photo of an old man` |
| FD Fig. 9 | `a photo of a rabbit` / `a photo of a duck` |
| FD Fig. 9 | `a photo of a skull` / `a photo of an old man` |
| FD Fig. 18 | `oil painting style, a car` / `oil painting style, a sports stadium` |
| FD Fig. 18 | `oil painting style, a heart` / `oil painting style, the grand canyon` |
| FD Fig. 18 | `a watercolor of a dog` / `a watercolor of new york city` |
| FD Fig. 18 | `a photo of a teddy bear` / `a photo of a bazaar` |

## `inverse_hybrid` — high slot only (the low slot is a reference image)

| citation | prompt |
|---|---|
| FD Fig. 1 | `a photo of a cat` |
| FD Fig. 1 | `an oil painting of a sunset` |
| FD Fig. 1 | `a watercolor of Duomo di Milano` |
| FD Fig. 8 | `a photo of a leopard` |
| FD Fig. 8 | `a photo of a lightbulb` |
| FD Fig. 8 | `a photo of waterfalls` |
