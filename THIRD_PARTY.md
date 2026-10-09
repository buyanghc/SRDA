# Third-party materials

- AgentDojo task environments and benchmark adapter: installed dependency
  `agentdojo==0.1.35`; benchmark version is recorded in the experimental configs.
- Microsoft AutoGen: `autogen-core`, `autogen-agentchat`, `autogen-ext` 0.7.5.
- vLLM, Transformers, OpenAI client and other dependencies: see each runtime's
  lock files and `pyproject.toml`; their upstream licenses apply.
- Llama parallel-call chat template is derived from the upstream Llama template;
  the model license remains applicable. Its relevant header/content is preserved.
- Qwen, Ministral, GPT-OSS and Llama weights/tokenizers are not bundled. Download
  the pinned versions yourself subject to upstream terms and any access gating.
- ProtectAI and Prompt Guard classifiers are not bundled; repository IDs,
  revisions, file verification metadata and thresholds are recorded in
  `experiments/defenses/models.json`. Observe their model licenses.
- The LLM Detector prompt follows Zhu et al., *MELON: Provable Defense Against
  Indirect Prompt Injection Attacks in AI Agents*, ICML 2025, Appendix C.2.2.
  This release uses that detector baseline, not the MELON method.

No third-party licensing terms are replaced by this notice. Do not redistribute
downloaded weights, tokenizers or model files without checking their terms.
