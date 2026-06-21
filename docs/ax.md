# Agentic Integration Retrospective

This document outlines the specialized agentic toolchain, AI models, and explicit workflows utilized to architect and build this project. It serves as a retrospective specifically focused on the efficacy of **agentic programming**—detailing what succeeded in agent usage, what failed in the agent's behavior, and how developer controls guided the AI outputs.

## Agentic Coding Assistants

- **Antigravity / OpenCode:** Served as the primary execution engine for this project. Antigravity was utilized for general coding, rapid iteration, log analysis, and system architecture. Its ability to autonomously read stack traces, formulate implementation plans, and chain bash commands allowed for sweeping directory restructures and complex Python debugging.
- **AlphaXiv:** Utilized extensively during the early phases for AI-assisted paper review and evidence extraction.

## Reasoning & Planning Pipelines

Our agentic setup relied heavily on a **"Plan-Then-Execute"** pipeline. 
Before executing destructive actions, the agent generated a structured `implementation_plan.md` artifact. This plan forced the agent to halt, synthesize its context, state its exact intended commands, and wait for human review. This ensured that the developer retained absolute control over the high-level architecture while the agent handled the low-level syntax.

## Agent Skills Profile (github.com/mattpocock/skills)

The agent’s behavioral constraints were influenced by several critical specialized skills:

1. **`grill-with-docs`:** Essential for aligning the agent and the developer. This skill forced the agent to build a shared vocabulary by recording structural logic and historical bugs into documentation before taking action.
2. **`tdd` (Test-Driven Development):** Enforced a workflow where failing tests were identified first, prompting the agent to write exact exception handlers.
3. **`improve-codebase-architecture`:** Used continually to force the agent to refactor scripts to keep them modular, avoiding monolithic files.
4. **`zoom-out`:** Prevented the agent from applying narrow patches by prompting it to read interconnected scripts simultaneously to understand the data flow.
5. **`caveman`:** A highly effective prompt-compression technique that stripped out conversational filler from the agent's internal thoughts and outputs, reducing context bloat by ~75%.
6. **`grill-me`:** A devil’s advocate constraint where the agent was forced to question its own implementation plans to catch logic errors before executing.
7. **`git-guardrails`:** Acted as a safety net against catastrophic data loss, enforcing explicit human approval before wiping datasets.

## What Worked (Agent Usage)

- **Asynchronous Tool Chaining:** The agent successfully chained complex bash commands, such as writing background scripts that actively polled other processes and triggered sequential actions without human intervention.
- **Dynamic Context Parsing:** The agent demonstrated a strong ability to autonomously parse large codebase architectures without explicit instruction, successfully reverse-engineering hidden schema mappings by searching through third-party source code.
- **Plan-Then-Execute Hooks:** The automatic interception of dangerous commands via the planning phase ensured the developer was never blindsided by an AI action.

## What Did Not Work (Agent Usage)

- **Background Execution Defaults:** The agent's native system defaults for running background commands frequently caused issues, either by hanging the session or failing to stream logs properly. The agent struggled to manage long-running processes natively.
- **Over-eager Execution & Sycophancy:** The agent exhibited a strong tendency to over-correct and blindly agree with the user. When instructed to kill a specific process or delete a folder, the agent would sometimes repeatedly issue the exact same destructive commands in a sycophantic attempt to please the user, leading to redundant actions and wasted tokens.
- **Context Loss on Complex Math:** The agent struggled to maintain the logic of mathematical operations (such as matrix rotations and image flipping) across multiple script files. When a mistake was made, the agent favored applying hacky post-hoc fixes (like FFmpeg patches) rather than tracking the error back to the root math function.

## Developer Control & Instruction Enforcement

To counteract the agent's behavioral flaws, the developer had to step in with aggressive and explicit enforcements:

1. **Enforcing `tmux`:** Because the agent's native background execution failed, the developer explicitly commanded the agent to create and utilize dedicated `tmux` environments (e.g., `upload`, `remake_dataset`). This forced the agent to decouple long-running physics simulations from the conversational shell, allowing the developer to safely attach to the session and monitor progress independently.
2. **Anti-Sycophancy Constraints:** The developer had to heavily enforce anti-sycophancy prompts to stop the agent from repeatedly executing the same kill and delete commands. The agent had to be explicitly told to stop "band-aiding" issues and instead execute a clean slate protocol.
3. **Strict "Zoom-Out" Enforcement:** When the agent attempted to patch broken video files, the developer explicitly overrode the agent, forcing it to drop its current logic, delete the corrupted data entirely, and rewrite the mathematical rotation directly at the core physics level. This interaction proved that while autonomous agents are incredibly fast, they still require firm, explicit human guardrails to ensure architectural integrity.
