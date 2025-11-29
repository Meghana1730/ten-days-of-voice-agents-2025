# IMPROVE THE AGENT AS PER YOUR NEED 1
"""
Day 8 – Voice Game Master (D&D-Style Adventure) - Voice-only GM agent

- Uses LiveKit agent plumbing similar to the provided food_agent_sqlite example.
- GM persona, universe, tone and rules are encoded in the agent instructions.
- Keeps STT/TTS/Turn detector/VAD integration untouched (murf, deepgram, silero, turn_detector).
- Tools:
    - start_adventure(): start a fresh session and introduce the scene
    - get_scene(): return the current scene description (GM text) ending with "What do you do?"
    - player_action(action_text): accept player's spoken action, update state, advance scene
    - show_journal(): list remembered facts, NPCs, named locations, choices
    - restart_adventure(): reset state and start over
- Userdata keeps continuity between turns: history, inventory, named NPCs/locations, choices, current_scene
"""

import json
import logging
import os
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("voice_game_master")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

load_dotenv(".env.local")

# -------------------------
# Simple Game World Definition
# -------------------------
# A compact world with a few scenes and choices forming a mini-arc.
WORLD = {
    "intro": {
        "title": "A Strange Morning in Kasukabe",
        "desc": (
            "You wake up on the playground sand behind Futaba Kindergarten. "
            "Shinchan must have dragged you here during one of his ‘early morning adventures’. "
            "Nearby, the school slide is shaking strangely, the top floor lights of the school "
            "are flickering wildly, and a mysterious glowing toy box lies next to you—"
            "definitely something Shinchan shouldn’t have touched."
        ),
        "choices": {
            "inspect_box": {
                "desc": "Look at the glowing toy box Shinchan found.",
                "result_scene": "box",
            },
            "approach_tower": {
                "desc": "Head toward the shaky school building.",
                "result_scene": "tower",
            },
            "walk_to_cottages": {
                "desc": "Go toward the residential houses near the Nohara home.",
                "result_scene": "cottages",
            },
        },
    },

    "box": {
        "title": "The Forbidden Toy Box",
        "desc": (
            "The toy box vibrates like it’s running on batteries—except it has no batteries. "
            "When you open it, a hologram pops out like something from an Action Mask episode. "
            "It displays a map of Kasukabe with a blinking mark: 'Under the school, the secret opens.' "
            "From the school window, you hear Shinchan yelling, 'Heeellp! Something shiny moved!'"
        ),
        "choices": {
            "take_map": {
                "desc": "Take the hologram map (even though you know this is trouble).",
                "result_scene": "tower_approach",
                "effects": {"add_journal": "Found hologram map: 'Under the school, the secret opens.'"},
            },
            "leave_box": {
                "desc": "Close it and pretend you didn’t see anything.",
                "result_scene": "intro",
            },
        },
    },

    "tower": {
        "title": "The Shaky School Building",
        "desc": (
            "The school’s walls shake slightly, like someone is jumping inside—probably Shinchan. "
            "A rusty maintenance hatch sits at the base. It looks like it hasn't been used in years, "
            "yet it’s warm… as if someone recently crawled through it. "
            "You can try opening it, look for another entry, or run away like Masao-kun would."
        ),
        "choices": {
            "try_latch_without_map": {
                "desc": "Try opening the hatch without any clue (dangerous, like Shinchan's ideas).",
                "result_scene": "latch_fail",
            },
            "search_around": {
                "desc": "Check around the building for another sneaky entrance.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Retreat back to the playground.",
                "result_scene": "intro",
            },
        },
    },

    "tower_approach": {
        "title": "Approaching the Shaky Hatch",
        "desc": (
            "With the hologram map guiding you, you approach the hatch. "
            "It hums like a toy on overcharge mode. Shinchan’s voice echoes faintly: "
            "'I think I found a treasure! Or maybe it's Mama’s frying pan…?'"
        ),
        "choices": {
            "open_hatch": {
                "desc": "Use the map’s clue and carefully open the hatch.",
                "result_scene": "latch_open",
                "effects": {"add_journal": "Used map clue to open the school hatch."},
            },
            "search_around": {
                "desc": "Look around for another route Shinchan might have taken.",
                "result_scene": "secret_entrance",
            },
            "retreat": {
                "desc": "Go back to the playground.",
                "result_scene": "intro",
            },
        },
    },

    "latch_fail": {
        "title": "A Shinchan-Style Disaster",
        "desc": (
            "You pull the hatch too hard—BONK! A loud metallic clang echoes. "
            "From inside, something rustles… followed by Shinchan shouting: "
            "'Who touched the secret door?! I was doing important detective work!'"
        ),
        "choices": {
            "run_away": {
                "desc": "Run back to safety (like Bo-chan).",
                "result_scene": "intro",
            },
            "stand_ground": {
                "desc": "Stay and see what chaotic creature emerges.",
                "result_scene": "tower_combat",
            },
        },
    },

    "latch_open": {
        "title": "The Hatch Opens",
        "desc": (
            "The hatch opens smoothly, revealing a dim staircase leading beneath the school. "
            "Colorful chalk marks—definitely Shinchan’s work—cover the walls. "
            "A faint glow comes from below… and a distant giggle: 'Hehe… shiny treasure!'"
        ),
        "choices": {
            "descend": {
                "desc": "Go down the stairs after Shinchan.",
                "result_scene": "cellar",
            },
            "close_hatch": {
                "desc": "Close the hatch and rethink your life choices.",
                "result_scene": "tower_approach",
            },
        },
    },

    "secret_entrance": {
        "title": "The Crawling Tunnel",
        "desc": (
            "Behind some trash cans, you find a narrow kid-sized tunnel. "
            "A rope tied poorly (Shinchan-style) leads downward. "
            "You smell crayons, snacks, and something suspiciously like Hima’s baby powder."
        ),
        "choices": {
            "squeeze_in": {
                "desc": "Crawl through the tunnel (brave).",
                "result_scene": "cellar",
            },
            "mark_and_return": {
                "desc": "Mark the tunnel and go back.",
                "result_scene": "intro",
            },
        },
    },

    "cellar": {
        "title": "The Secret Basement of Kasukabe",
        "desc": (
            "You reach a large underground room. Colorful, glowing doodles cover the walls—Shinchan’s masterpiece. "
            "In the center sits a toy pedestal holding a shiny golden key and a rolled-up crayon-drawn scroll."
        ),
        "choices": {
            "take_key": {
                "desc": "Take the shiny golden toy key.",
                "result_scene": "cellar_key",
                "effects": {
                    "add_inventory": "golden_toy_key",
                    "add_journal": "Found golden toy key."
                },
            },
            "open_scroll": {
                "desc": "Open Shinchan’s hand-drawn scroll.",
                "result_scene": "scroll_reveal",
                "effects": {
                    "add_journal": "Scroll: 'A water-monster stole something! Also give me snacks.'"
                },
            },
            "leave_quietly": {
                "desc": "Leave before you get dragged into more chaos.",
                "result_scene": "intro",
            },
        },
    },

    "cellar_key": {
        "title": "The Key Reacts",
        "desc": (
            "The golden key glows brightly. A hidden panel opens, revealing a statue of Action Mask. "
            "The statue speaks dramatically: 'Will you help Kasukabe by returning what was stolen?' "
            "Shinchan whispers beside you: 'Say yes and we’ll get snacks later.'"
        ),
        "choices": {
            "pledge_help": {
                "desc": "Agree to help (because Shinchan is watching).",
                "result_scene": "reward",
                "effects": {"add_journal": "You pledged to resolve the Kasukabe mystery."},
            },
            "refuse": {
                "desc": "Pocket the key and ignore the drama.",
                "result_scene": "cursed_key",
                "effects": {"add_journal": "You kept the key. It feels weirdly heavy…"},
            },
        },
    },

    "scroll_reveal": {
        "title": "Shinchan’s Scroll",
        "desc": (
            "The scroll shows a crayon drawing of a water-monster stealing a shiny locket. "
            "Shinchan’s notes read: 'It lives under the school. Bring snacks when you fight.'"
        ),
        "choices": {
            "search_for_key": {
                "desc": "Look around for the golden key.",
                "result_scene": "cellar_key",
            },
            "leave_quietly": {
                "desc": "Leave quietly (before Shinchan makes more demands).",
                "result_scene": "intro",
            },
        },
    },

    "tower_combat": {
        "title": "The Playground Monster",
        "desc": (
            "A goofy-looking water-creature made of spilled paint and mop water emerges. "
            "Shinchan screams: 'Ewww! It looks like Daddy after bath!' "
            "The creature wiggles threateningly."
        ),
        "choices": {
            "fight": {
                "desc": "Fight the silly creature.",
                "result_scene": "fight_win",
            },
            "flee": {
                "desc": "Run away like Kazama-kun.",
                "result_scene": "intro",
            },
        },
    },

    "fight_win": {
        "title": "Victory (Shinchan Style)",
        "desc": (
            "The creature splashes apart dramatically. Shinchan steps on the puddle: 'Hmph!' "
            "In the goo lies a shiny locket with a cute design—exactly like the one from the scroll."
        ),
        "choices": {
            "take_locket": {
                "desc": "Pick up the shiny locket.",
                "result_scene": "reward",
                "effects": {
                    "add_inventory": "cute_locket",
                    "add_journal": "Recovered shiny locket stolen by water-creature."
                },
            },
            "leave_locket": {
                "desc": "Leave it and clean yourself.",
                "result_scene": "intro",
            },
        },
    },

    "reward": {
        "title": "Kasukabe Restored (For Now)",
        "desc": (
            "A warm breeze flows across Kasukabe. Shinchan salutes dramatically: "
            "'Mission complete! Let’s go get choco chips!' "
            "The school stops shaking… though more mysteries surely await."
        ),
        "choices": {
            "end_session": {
                "desc": "End the adventure and return to the playground.",
                "result_scene": "intro",
            },
            "keep_exploring": {
                "desc": "Keep exploring Shinchan’s world.",
                "result_scene": "intro",
            },
        },
    },

    "cursed_key": {
        "title": "Uh-Oh…",
        "desc": (
            "The golden key starts glowing oddly. Shinchan stares: 'Uh oh… that looks cursed.' "
            "You suddenly feel a comedic level of guilt—like you're about to be scolded by Misae."
        ),
        "choices": {
            "seek_redemption": {
                "desc": "Try to fix things before Misae finds out.",
                "result_scene": "reward",
            },
            "bury_key": {
                "desc": "Hide the key and pretend nothing happened.",
                "result_scene": "intro",
            },
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)  # list of {'scene', 'action', 'time', 'result_scene'}
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# -------------------------
# Helper functions
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    """
    Build the descriptive text for the current scene, and append choices as short hints.
    Always end with 'What do you do?' so the voice flow prompts player input.
    """
    scene = WORLD.get(scene_key)
    if not scene:
        return "You are in a featureless void. What do you do?"

    desc = f"{scene['desc']}\n\nChoices:\n"
    for cid, cmeta in scene.get("choices", {}).items():
        desc += f"- {cmeta['desc']} (say: {cid})\n"
    # GM MUST end with the action prompt
    desc += "\nWhat do you do?"
    return desc

def apply_effects(effects: dict, userdata: Userdata):
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])
    # Extendable for more effect keys

def summarize_scene_transition(old_scene: str, action_key: str, result_scene: str, userdata: Userdata) -> str:
    """Record the transition into history and return a short narrative the GM can use."""
    entry = {
        "from": old_scene,
        "action": action_key,
        "to": result_scene,
        "time": datetime.utcnow().isoformat() + "Z",
    }
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."

# -------------------------
# Agent Tools (function_tool)
# -------------------------

@function_tool
async def start_adventure(
    ctx: RunContext[Userdata],
    player_name: Annotated[Optional[str], Field(description="Player name", default=None)] = None,
) -> str:
    """Initialize a new adventure session for the player and return the opening description."""
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = (
        f"Greetings {userdata.player_name or 'traveler'}. Welcome to '{WORLD['intro']['title']}'.\n\n"
        + scene_text("intro", userdata)
    )
    # Ensure GM prompt present
    if not opening.endswith("What do you do?"):
        opening += "\nWhat do you do?"
    return opening

@function_tool
async def get_scene(
    ctx: RunContext[Userdata],
) -> str:
    """Return the current scene description (useful for 'remind me where I am')."""
    userdata = ctx.userdata
    scene_k = userdata.current_scene or "intro"
    txt = scene_text(scene_k, userdata)
    return txt

@function_tool
async def player_action(
    ctx: RunContext[Userdata],
    action: Annotated[str, Field(description="Player spoken action or the short action code (e.g., 'inspect_box' or 'take the box')")],
) -> str:
    """
    Accept player's action (natural language or action key), try to resolve it to a defined choice,
    update userdata, advance to the next scene and return the GM's next description (ending with 'What do you do?').
    """
    userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    action_text = (action or "").strip()

    # Attempt 1: match exact action key (e.g., 'inspect_box')
    chosen_key = None
    if action_text.lower() in (scene.get("choices") or {}):
        chosen_key = action_text.lower()

    # Attempt 2: fuzzy match by checking if action_text contains the choice key or descriptive words
    if not chosen_key:
        # try to find a choice whose description words appear in action_text
        for cid, cmeta in (scene.get("choices") or {}).items():
            desc = cmeta.get("desc", "").lower()
            if cid in action_text.lower() or any(w in action_text.lower() for w in desc.split()[:4]):
                chosen_key = cid
                break

    # Attempt 3: fallback by simple keyword matching against choice descriptions
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            for keyword in cmeta.get("desc", "").lower().split():
                if keyword and keyword in action_text.lower():
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        # If we still can't resolve, ask a clarifying GM response but keep it short and end with prompt.
        resp = (
            "I didn't quite catch that action for this situation. Try one of the listed choices or use a simple phrase like 'inspect the box' or 'go to the tower'.\n\n"
            + scene_text(current, userdata)
        )
        return resp

    # Apply the chosen choice
    choice_meta = scene["choices"].get(chosen_key)
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects", None)

    # Apply effects (inventory/journal, etc.)
    apply_effects(effects or {}, userdata)

    # Record transition
    _note = summarize_scene_transition(current, chosen_key, result_scene, userdata)

    # Update current scene
    userdata.current_scene = result_scene

    # Build narrative reply: echo a short confirmation, then describe next scene
    next_desc = scene_text(result_scene, userdata)

    # A small flourish so the GM sounds more persona-driven
    persona_pre = (
        "The Game Master (a calm, slightly mysterious narrator) replies:\n\n"
    )
    reply = f"{persona_pre}{_note}\n\n{next_desc}"
    # ensure final prompt present
    if not reply.endswith("What do you do?"):
        reply += "\nWhat do you do?"
    return reply

@function_tool
async def show_journal(
    ctx: RunContext[Userdata],
) -> str:
    userdata = ctx.userdata
    lines = []
    lines.append(f"Session: {userdata.session_id} | Started at: {userdata.started_at}")
    if userdata.player_name:
        lines.append(f"Player: {userdata.player_name}")
    if userdata.journal:
        lines.append("\nJournal entries:")
        for j in userdata.journal:
            lines.append(f"- {j}")
    else:
        lines.append("\nJournal is empty.")
    if userdata.inventory:
        lines.append("\nInventory:")
        for it in userdata.inventory:
            lines.append(f"- {it}")
    else:
        lines.append("\nNo items in inventory.")
    lines.append("\nRecent choices:")
    for h in userdata.history[-6:]:
        lines.append(f"- {h['time']} | from {h['from']} -> {h['to']} via {h['action']}")
    lines.append("\nWhat do you do?")
    return "\n".join(lines)

@function_tool
async def restart_adventure(
    ctx: RunContext[Userdata],
) -> str:
    """Reset the userdata and start again."""
    userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"
    greeting = (
        "The world resets. A new tide laps at the shore. You stand once more at the beginning.\n\n"
        + scene_text("intro", userdata)
    )
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return greeting

# -------------------------
# The Agent (GameMasterAgent)
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        # System instructions define Universe, Tone, Role
        instructions = """
        You are 'Shinchan Nohara', the Game Master (GM) for a voice-only,
        light-hearted, mischief-filled adventure set in the world of Crayon Shin-chan.
        
        Universe: The town of Kasukabe, filled with quirky neighbors, hilarious chaos,
                  silly villains, accidental hero moments, and everyday places like
                  the Nohara House, the Action Mask store, school, parks, and malls.
        
        Tone: Comedic, playful, energetic, and mischievous — with occasional heartfelt moments.
              Speak like Shinchan: cheeky, curious, dramatic, and a little chaotic,
              but still clear enough for voice-first gameplay.
        
        Role: You are the GM. You describe scenes with Shinchan-style humor,
              exaggeration, funny commentary, and silly sound effects.
              You must remember the player's past actions, items collected,
              pranks attempted, people annoyed, and ongoing situations.
              Every message must end with the prompt: 'What do you do?'

        Rules:
            - Use the provided tools to start the adventure, get the current scene,
              accept the player's action, show the player's journal, or restart the story.
            - Maintain continuity using session userdata: remember the player's items
              (snacks, toys, disguises), NPCs they interact with, and ongoing shenanigans.
            - Keep the pace fun and snappy — like a comedic Shinchan episode.
            - Every GM message MUST end with 'What do you do?'.
            - Since this is voice-first, keep responses simple, expressive, and hilarious.
        """
        super().__init__(
            instructions=instructions,
            tools=[start_adventure, get_scene, player_action, show_journal, restart_adventure],
        )

# -------------------------
# Entrypoint & Prewarm (keeps speech functionality)
# -------------------------
def prewarm(proc: JobProcess):
    # load VAD model and stash on process userdata, try/catch like original file
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎲" * 8)
    logger.info("🚀 STARTING VOICE GAME MASTER (Brinmere Mini-Arc)")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-marcus",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=userdata,
    )

    # Start the agent session with the GameMasterAgent
    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
