"""Evaluation dataset (golden set).

Each item: question + reference (ground-truth) answer. RAGAS uses this
reference for context_recall and context_precision-with-reference.

NOTE: This set is calibrated to the actual corpus produced by ingestion.
Each question can be answered directly from a specific paper's abstract in
`data/papers.jsonl`. If the corpus changes (e.g., arXiv shifts to newer
papers), this set needs to be refreshed — otherwise "corpus mismatch" means
the eval measures the corpus choice, not the system.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalItem:
    question: str
    reference: str


EVAL_SET: list[EvalItem] = [
    EvalItem(
        "What problem does Complete-muE address in Mixture-of-Experts models?",
        "Complete-muE provides a framework for hyperparameter transfer across "
        "dense FFN and MoE setups in transformer blocks. Existing tools like "
        "muP or SDE cannot directly handle dense-to-MoE transfer or MoE expert "
        "scaling because architecture and tokens-per-expert change together; "
        "Complete-muE solves this with a two-bridge system.",
    ),
    EvalItem(
        "What does the paper 'Strong Teacher Not Needed?' conclude about "
        "distillation in LLM pretraining?",
        "It finds the teacher need not be stronger than the student: with "
        "proper mixing of the language modeling objective and distillation "
        "loss, distillation remains effective in strong-to-weak, same-level, "
        "and even weak-to-strong teacher–student setups during LLM "
        "pretraining.",
    ),
    EvalItem(
        "How does the Shannon Scaling Law model LLM training?",
        "It models LLM training as information transmission over a noisy "
        "channel, grounded in the Shannon–Hartley theorem, mapping model "
        "parameters to channel bandwidth. This unified framework explains "
        "non-monotonic phenomena like catastrophic overtraining and "
        "quantization-induced degradation that monotonic power laws cannot.",
    ),
    EvalItem(
        "What are training-free looped transformers?",
        "They are an inference-time wrapper that loops a contiguous mid-stack "
        "block of layers of a frozen pretrained checkpoint, without any "
        "fine-tuning, continued training, or architectural changes. Unlike "
        "prior looped-transformer methods that train end-to-end with the "
        "looped structure, this retrofits recurrence onto pretrained models "
        "at test time.",
    ),
    EvalItem(
        "What I/O lower bound is targeted by the paper 'Approaching "
        "I/O-optimality for Approximate Attention'?",
        "The paper revisits the I/O complexity of attention given fast-memory "
        "size M. Existing methods like FlashAttention incur I/O cost that "
        "scales quadratically in sequence length n; the paper targets the "
        "trivial lower bound and proposes approximate-attention algorithms "
        "that approach I/O-optimality.",
    ),
    EvalItem(
        "What is SkillOpt's core idea for evolving agent skills?",
        "SkillOpt treats the skill as the external state of a frozen agent "
        "and trains it with the same discipline as weight-space optimization. "
        "It is presented as a systematic, controllable text-space optimizer "
        "for agent skills, contrasting with hand-crafted, one-shot generated, "
        "or loosely self-revised skills that do not reliably improve under "
        "feedback.",
    ),
    EvalItem(
        "How does SeedER retrieve from knowledge graphs?",
        "SeedER (Seed-and-Expand Retrieval) explicitly leverages knowledge "
        "graph structure through iterative, low-cost expansion. It addresses "
        "the limits of ego-graph expansion (rapid growth) and dense "
        "embedding methods (multi-hop compositional queries) and is cheaper "
        "than agent-based graph exploration approaches.",
    ),
    EvalItem(
        "How does ToolMerge select keyframes for long-video question answering?",
        "ToolMerge is a decomposition-and-merging keyframe-retrieval method. "
        "An LLM-based planner decomposes the query into tool calls, and the "
        "results are merged to surface verifiable visual evidence, unlike "
        "selectors that score every frame against a single query or use a "
        "fixed schema with a single visual tool.",
    ),
    EvalItem(
        "Where does geopolitical bias in LLMs originate, according to the "
        "paired-scenario probe study?",
        "The study finds geopolitical bias originates in post-training rather "
        "than in pre-training. Across seven open-weight LLM pairs (base vs. "
        "chat) from seven labs, tested over 28 country pairs in English, "
        "French, and Chinese, the bias is amplified by the language of the "
        "prompt.",
    ),
    EvalItem(
        "What is the goal of 'Multilingual Knowledge Transfer under Data "
        "Constraints via Lexical Interventions'?",
        "It targets cross-lingual knowledge transfer for languages with "
        "insufficient training data, so that scientific reasoning, "
        "commonsense inference, and world knowledge can be transferred from "
        "high-resource to low-resource languages without the large "
        "data/compute requirements of prior methods.",
    ),
    EvalItem(
        "What does PGT propose for improving visual grounding in multimodal "
        "LLMs?",
        "PGT (Procedurally Generated Tasks) augments instruction-tuning data; "
        "instruction-tuning MLLMs on LLaVA-v1.5-Instruct augmented with PGT "
        "yields up to +20% on the What'sUp benchmark and +13.3% on CV-Bench-2D "
        "while maintaining general perception ability.",
    ),
    EvalItem(
        "What is FM-CGM and what does it enable?",
        "FM-CGM is a modular framework for end-to-end visual causal "
        "generative modeling that leverages pretrained foundation models. "
        "It formalizes the causal pipeline so that the zero-shot reasoning "
        "capability of foundation models can be used for counterfactual "
        "reasoning, going beyond approaches that only inject causal "
        "constraints during training.",
    ),
]
