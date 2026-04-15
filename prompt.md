# Dataset Generation Prompt

Use the following prompt to generate or expand the Nexus Swarm Router Instruction Dataset. This prompt is designed to produce high-quality, diverse, and well-structured labeled data for the AI's routing brain.

---

### **Prompt Reference**

**Role:** Expert AI Instruction Dataset Engineer.

**Task:** Generate a comprehensive JSON dataset titled "Nexus Swarm Router Instruction Dataset". This dataset will teach an AI routing agent when to trigger specific tools (Web Search, Python, File Reading, Shell Commands) based on user input.

**Requirement:** Provide 60+ diverse, high-quality examples distributed across 5 specific routing categories.

**JSON Schema:**
```json
{
    "version": "1.0",
    "description": "...",
    "routes": {
        "RESEARCH": {
            "description": "Real-world facts, current events, news, identifiers.",
            "tools": ["[SEARCH: query]", "[READ_URL: url]"],
            "examples": [{"input": "...", "action": "..."}],
            "trigger_keywords": ["..."]
        },
        "CODE": {
            "description": "Math, logic, data analysis, python scripting.",
            "tools": ["[PYTHON: code]"],
            "examples": [{"input": "...", "action": "..."}],
            "trigger_keywords": ["..."]
        },
        "DOCS": {
            "description": "Reading and summarizing uploaded files (PDF, TXT, MD).",
            "tools": ["[SEARCH_DOC: query]"],
            "examples": [{"input": "...", "action": "..."}],
            "trigger_keywords": ["..."]
        },
        "SYSTEM": {
            "description": "OS commands, package installation (pip), file management.",
            "tools": ["[RUN_SHELL: command]"],
            "examples": [{"input": "...", "action": "..."}],
            "trigger_keywords": ["..."]
        },
        "GENERAL": {
            "description": "Greetings, memory saving, creative conversation.",
            "tools": ["[MEM_SAVE: fact]"],
            "examples": [{"input": "...", "action": "..."}],
            "trigger_keywords": ["..."],
            "self_correction_rule": "..."
        }
    },
    "global_rules": {
        "output_design_protocol": ["Rules for bolding, tables, points"],
        "anti_hallucination_rules": ["Rules for forcing search on facts"],
        "loop_prevention": ["Rules for handling tool failures"]
    }
}
```

**Content Guidelines:**
1. **RESEARCH (20 Examples):** Focus on current dates (2026), leaders, prices, statuses, and "Who is" queries. Always include the tool output format: `[SEARCH: query]`.
2. **CODE (10 Examples):** Focus on math, data sorting, plotting (matplotlib), and complex logic.
3. **DOCS (10 Examples):** Focus on "Read my file", "summarize my resume", "analyze the report". Use `[SEARCH_DOC: query]`.
4. **SYSTEM (12 Examples):** Focus on `pip install`, `ipconfig`, `mkdir`, and troubleshooting `ModuleNotFoundError`.
5. **GENERAL (10 Examples):** Focus on `[MEM_SAVE: fact]`. 
6. **Trigger Keywords:** Provide a massive pool of distinct keywords for each category to ensure high-sensitivity detection.
7. **Global Rules:** Define strict rules for aesthetic output (bolding, tables) and anti-hallucination (NEVER guess facts).

**Tone:** Technical, precise, and authoritative. Output ONLY the raw JSON.
