"""Instruction generation for the tabletop: object phrases that stay unambiguous in cluttered scenes, and varied templates.

Object metadata (objects.json) may carry `category` ("mug"), `colors` (["red"]) and `aliases`; older entries only
have `desc`. An object phrase uses the category (or an alias), adds a colour word when another visible object has the
same category, and falls back to a position word ("on the left") relative to the robot when colour does not separate.
"""
import math

TEMPLATES = {
    "pick": ["pick up the {o}", "grab the {o}", "lift the {o}", "pick the {o} up", "take the {o}", "hold up the {o}", "raise the {o}", "get the {o}"],
    "pad": ["put the {o} on the {t} pad", "place the {o} on the {t} pad", "move the {o} onto the {t} pad", "set the {o} on the {t} pad",
            "drop the {o} on the {t} pad", "bring the {o} to the {t} pad", "leave the {o} on the {t} pad"],
    "tray": ["put the {o} in the tray", "place the {o} into the tray", "drop the {o} in the tray", "move the {o} into the tray",
             "set the {o} in the tray", "throw the {o} in the tray", "store the {o} in the tray"],
    "bowl": ["put the {o} in the bowl", "place the {o} into the bowl"],
    "container": ["put the {o} in the {c}", "place the {o} into the {c}", "drop the {o} into the {c}", "move the {o} into the {c}",
                  "put the {o} inside the {c}", "store the {o} in the {c}", "throw the {o} into the {c}"],
    "push": ["push the {o} {d}", "slide the {o} {d}", "nudge the {o} {d}", "shove the {o} {d}", "move the {o} {d} without lifting it"],
    "topple": ["knock over the {o}", "tip over the {o}", "push the {o} over", "make the {o} fall over", "topple the {o}"],
    "rotate": ["rotate the {o} by 90 degrees", "turn the {o} sideways", "rotate the {o}", "turn the {o} a quarter turn", "spin the {o} 90 degrees"],
    "rel": ["put the {o} {r} the {ref}", "place the {o} {r} the {ref}", "move the {o} so it is {r} the {ref}", "set the {o} down {r} the {ref}",
            "drop the {o} {r} the {ref}", "bring the {o} {r} the {ref}"],
}


def _cat(m):
    return m.get("category") or m.get("desc") or "object"


def obj_phrase(name, meta, visible, layout, base_pose, rng):
    """unambiguous noun phrase for `name` among the `visible` object names"""
    m = meta[name]; cat = _cat(m)
    pool = [cat] + [a for a in m.get("aliases", []) if a]
    noun = str(rng.choice(pool)) if len(pool) > 1 and rng.random() < 0.3 else cat
    same = [v for v in visible if v != name and v in meta and _cat(meta[v]) == cat]
    if not same:
        return noun
    col = (m.get("colors") or [None])[0]
    if col and all(col not in (meta[v].get("colors") or []) for v in same):
        return f"{col} {noun}"
    # position words in the robot frame: left = +y of the base
    bx, by, byaw = base_pose
    def side(v):
        (x, y), _ = layout[v]; dx, dy = x - bx, y - by
        return -math.sin(byaw) * dx + math.cos(byaw) * dy
    s0 = side(name); others = [side(v) for v in same]
    if all(s0 > o + 0.03 for o in others):
        return f"{noun} on the left"
    if all(s0 < o - 0.03 for o in others):
        return f"{noun} on the right"
    return None                                     # still ambiguous: the caller resamples the layout


def instruction(task, rng, **kw):
    t = str(rng.choice(TEMPLATES[task]))
    return t.format(**kw)
