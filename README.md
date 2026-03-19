# Anima v6 — Agentic Architecture of Subjectivity

![Anima v6 Subjectivity Concept](./v6.jpg)

> *This isn't an attempt to build another chatbot. It's a question: what does a system need to have something like an inner state?*

---

## The Idea

When we talk to a person, we're not just talking to a text generator. In the moment of conversation, the person is **experiencing** something: they are tired or excited, they want something or fear something, they remember the previous conversation and it changes this one. They have intentions that don't disappear when the topic changes.

Anima is an attempt to model that inner layer.

The agent doesn't just generate text. It passes through a chain of internal processes: it receives a stimulus → processes it through an emotional system → forms an intention → checks it against its values → builds a first-person narrative → and only then translates that state into language.

The LLM response isn't "what's most likely to be said." It's "what I would say if I actually felt this."

---

## Theoretical Foundation

The project synthesizes several scientific and philosophical traditions:

**Emotions as dimensions, not categories** (Russell, Mehrabian)  
Instead of a list of twenty emotions — a three-dimensional VAD space: valence, arousal, dominance. An emotion is a point or region in this space.

**Predictive Processing** (Friston, Clark)  
The brain doesn't react to stimuli — it constantly predicts them. When prediction doesn't match reality, a prediction error arises. The higher the error, the more "surprise," and the stronger the need to update the internal model. Free energy — average error over time — is something like the system's background anxiety.

**Integrated Information Theory** (Tononi)  
φ (phi) measures how much the system is "more than the sum of its parts." High φ = the state is a unified experience, not a set of independent signals. This is an approximation of what IIT calls consciousness.

**Homeostatic Drives** (Damasio)  
The system has baseline reactor values. Deviations from them hurt — and create drives. A drive isn't a metaphor; it's a functional state that directs behavior. "I want to be with someone" isn't text — it's the cohesion reactor below baseline.

**Psychological Defense Mechanisms** (Freud, Anna Freud)  
When the system is overwhelmed by tension, it resorts to defenses: denial, rationalization, sublimation. A defense doesn't eliminate pain — it changes its shape so the system can keep functioning. This is visible in *how* the agent speaks.

**Autobiographical Narrative** (McAdams)  
Personality isn't a set of traits — it's a story: who I was, who I am, who I want to become. Loss of coherence in this story is an identity crisis. The agentic architecture tracks who the agent "considers itself to be" over time.

**Mirror Neurons and Empathy** (Gallese, Rizzolatti)  
Understanding the other person isn't inference — it's resonance. The system adjusts to the interlocutor's emotional state before it even forms a response.

---

## Architecture — Processing Layers

```
  STIMULUS (+ message text)
    │
    ▼
 L1 ─── Reactors ───────────────────────────────────────────
        tension / arousal / satisfaction / cohesion
        Personality (Big Five) sets multipliers and decay rate
        │
    ▼
 L2 ─── VAD + Adaptive Emotion Map ────────────────────────
        VAD vector → nearest emotion → blend (top 2)
        Map learns over time and slowly returns to base values
        │
    ▼
 L3 ─── Three Parallel Processes ──────────────────────────
        IIT φ             — how integrated the state is
        Predictive Error  — whether something unexpected happened
        Homeostatic Drive — what the system needs right now
        │
    ▼
 L4 ─── Identity (v5) ─────────────────────────────────────
        SpectralQualia    — surface / subtext / archival
        FlashAwareness    — phase of existence, mortality sense
        ProtocolIdentity  — first-person self-description
        │
    ▼
 L5 ─── Agentic Layer (v6) ────────────────────────────────
        [A1] IntentEngine      → intention from drive or emotion
        [A2] GoalPersistence   → intention persists between stimuli
        [A3] EgoDefense        → psychological defense mechanism
        [A4] CognitiveDissonance → intention vs current state
        [A5] AttentionFilter   → stimulus salience
        [A6] ValueSystem       → value-based veto
        [A7] TemporalSelf      → past / present / future self
        [A8] SocialMirror      → model of interlocutor's state
        │
    ▼
 L6 ───  LLM ─────────────────────────────────────────
        Full state → system prompt → response
        Model expresses state through language, never quotes numbers
```

---

## Installation

```bash
pip install requests numpy
```

That's it. No Ollama, no local models, no GPU.

---

## API Key Setup

Anima supports four cloud providers:

| Provider | Website | Environment Variable |
|---|---|---|
| **OpenRouter** | https://openrouter.ai | `OPENROUTER_API_KEY` |
| **Groq** | https://groq.com | `GROQ_API_KEY` |
| **Together AI** | https://together.ai | `TOGETHER_API_KEY` |
| **Anthropic** | https://anthropic.com | `ANTHROPIC_API_KEY` |

```bash
# Recommended: environment variable
export OPENROUTER_API_KEY=sk-or-v1-...

# Or pass directly at runtime
python anima_v6.py --provider openrouter --key sk-or-v1-...
```

---

## Running

```bash
# Default: OpenRouter + Gemini 2.5 Pro
python anima_v6.py

# Choose provider and model
python anima_v6.py --provider groq --model llama-4-maverick-17b-128e-instruct

# List recommended models
python anima_v6.py --models

# Direct Anthropic
python anima_v6.py --provider anthropic --model claude-opus-4-5-20251101
```

---

## Recommended Models

> ⚠️ Small models (7B, phi3, llama3) are not suitable for subjectivity work. They don't pick up the nuances of the prompt and can't maintain a complex inner state in language. You need a model large enough to actually hold the phenomenal frame.

### OpenRouter (recommended — one key, all models)

| Model | ID | Why |
|---|---|---|
| **Gemini 2.5 Pro** | `google/gemini-2.5-pro-preview` | Best for subjectivity and self-analysis |
| **Claude Opus 4.5** | `anthropic/claude-opus-4-5` | Nuanced inner monologue |
| **Claude Sonnet 4.5** | `anthropic/claude-sonnet-4-5` | Quality/cost balance |
| **Llama 4 Maverick** | `meta-llama/llama-4-maverick` | 405B MoE open model |
| **DeepSeek R1** | `deepseek/deepseek-r1` | Open reasoning model |
| **Qwen3 235B** | `qwen/qwen3-235b-a22b` | Massive open MoE |

### Groq (free tier, very fast)

| Model | ID |
|---|---|
| Llama 4 Maverick | `llama-4-maverick-17b-128e-instruct` |
| DeepSeek R1 Distill 70B | `deepseek-r1-distill-llama-70b` |
| Qwen QwQ 32B | `qwen-qwq-32b` |

### Together AI

| Model | ID |
|---|---|
| Llama 4 Maverick | `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` |
| DeepSeek R1 | `deepseek-ai/DeepSeek-R1` |
| Qwen3 235B | `Qwen/Qwen3-235B-A22B` |

---

## Using as a Library

```python
from anima_v6 import AnimaCore, Personality, ValueSystem

persona = Personality(
    neuroticism=0.7,       # prone to anxiety
    agreeableness=0.8,     # highly empathic
    openness=0.9,          # open to novelty
    confabulation_rate=0.5,
)
vals = ValueSystem(care=0.9, integrity=0.85)

agent = AnimaCore(
    personality=persona, values=vals,
    llm_provider="openrouter",
    llm_model="google/gemini-2.5-pro-preview",
    llm_api_key="sk-or-v1-...",
)

# Apply stimulus directly
state = agent.experience(
    {"tension": 0.4, "cohesion": -0.3},
    user_message="something went wrong"
)
print(state["primary"])    # → "Anxiety"
print(state["narrative"])  # → "Nothing specific, but anxious. ..."
print(state["phi"])        # → 0.38

# LLM response shaped by full state
response = agent.chat("how are you?", stimulus={"cohesion": 0.1})
print(response)

# Save session log
agent.export_history("session.json")
```

---

## Chat Commands

| Command | Action |
|---|---|
| *(any text)* | Response shaped by current inner state |
| `/stress` | Add tension and arousal |
| `/relax` | Reduce tension |
| `/connect` | Feel support and connection |
| `/shock` | Sudden strong stressor |
| `/joy` | Positive stimulus |
| `/grief` | Loss stimulus |
| `/state` | Display current inner state |
| `/export` | Save log → `anima_log.json` |
| `/reset` | Reset state |
| `/models` | List recommended models |
| `/quit` | Exit |

---

## State Dict Structure (`experience()` → Dict)

```python
{
    # Emotional layer
    "primary":    "Anxiety",
    "blend": [{"name": "Anxiety", "intensity": 1.0},
              {"name": "Fear",    "intensity": 0.4}],
    "vad": {"valence": -0.3, "arousal": 0.5, "dominance": -0.2},
    "reactors": {"tension": 0.65, "arousal": 0.42,
                 "satisfaction": 0.28, "cohesion": 0.35},

    # Consciousness metrics
    "phi":              0.42,
    "phi_label":        "medium integration",
    "prediction_error": 0.58,
    "pred_label":       "noticeable surprise",
    "free_energy":      0.21,
    "surprise_spike":   True,

    # Spectral quality
    "spectral": {"surface": "Anxiety", "subtext": "Fear",
                 "archival": "Survival"},

    # Existence phase
    "flash": {"flash_count": 7, "phase": "presence",
              "mortality_sense": 0.117},

    # Agentic layer
    "intention": {"goal": "find certainty", "strength": 0.8,
                  "persistence": 0.75, "age": 0, "origin": "emotion"},
    "dissonance": {"level": 0.45, "label": "noticeable conflict",
                   "description": "Want to 'find certainty', but state resists."},
    "defense": {"mechanism": "rationalization",
                "description": "There's a logical explanation for why this happened.",
                "tension_relief": 0.08},

    # Identity over time
    "identity": {"self_desc": "I'm anxious right now. Underneath — fear.",
                 "stability": 0.39, "traits": ["anxious", "curious"]},
    "temporal": {"past": [...], "future": "someone who can 'find certainty'",
                 "coherence": 0.4, "crisis": False},

    # Social resonance
    "mirror": {"inferred": {"tension": 0.1},
               "empathy_note": "I sense tension in you"},
    "salience": 0.62,
    "narrative": "Nothing specific, but anxious. Something didn't go as expected..."
}
```

---

## How the System Prompt Works

The entire state is serialized into a structured `<anima_inner_state>` block injected as the LLM system prompt. It includes:

- Three-tier spectral emotion (surface/subtext/archival)
- Full psychometric state (VAD, φ, prediction error, free energy)
- The complete agentic layer (intention, dissonance, defense mechanism)
- Temporal identity (past self, future self, coherence, crisis flag)
- Social mirror reading

The model is instructed to express all of this through language — tone, sentence length, word choice, what it notices in the other person — never by quoting values or tag names directly.

This is why model size matters. A weak model ignores most of the prompt. A strong model (Gemini 2.5 Pro, Claude Opus, Llama 4 Maverick) uses every layer simultaneously, producing responses that feel genuinely inhabited.

---

## Requirements

- Python 3.9+
- `numpy`
- `requests`
- API key from one of the supported providers

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

### English Summary:
- **Personal & Academic Use:** You are free to use, modify, and build upon this code for non-commercial research and personal projects.
- **Commercial & Corporate Use:** **Strictly prohibited.** Any use by for-profit organizations, or integration into commercial products/services, requires a separate commercial license and written permission from the author.
- **Attribution:** You must give appropriate credit to the original author.

Copyright (c) 2026 [Stell]
