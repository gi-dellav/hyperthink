# Archetype for HyperThink implementations

This documents contains base specifications that should be followed by libraries that implement the HyperThink scaffolding.

- They must be based on a pre-existing LLM library.
- They must be well integrated with the pre-existing LLM library.
- They must implement a single function that executes a query using the HyperThink scaffolding.
- They must implement a core class for managing the HyperThink scaffolding.
- They must allow operation logging (printing out tool calls, notes editing, etc).
- They must allow JSON checkpoints (save and load the current state of the scaffolding).
- They must allow to set a limit of inference iterations (if the limit is reached, output the current state).
- They must allow to change the used models (A and B).
- They must allow to change the reasoning effort of the models.
- They must allow to change the temperaturs and top_k of the models.
- They must allow to change the system prompts of the models.

- They must use by the default the standard system prompts (given in the *_PREAMBLE.md).
- They must use `{}` and `{}` formatting for the Review model's system prompt.
- They must use message history only for real messages, not for the intermediate steps managed by the HyperThink scaffolding.

- If possible, they should implement cost extimation.
- If possible, they should implement MCP support.
- If possible, they should implement Web Search support.
- If possible, they should implement a math solver tool, powered by computer algebric systems (ex. SymPy).

- They should implement a plan system, that allows to first split into a list of smaller tasks, and then run each task using the HyperThink scaffolding.

- They must set Deepseek V3.2 as the default Model A and Gemini 3 Flash as the default Model B.

- They must use assert instructions in order to avoid any impossible or unvalid code execution path.
