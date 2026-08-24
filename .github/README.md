# Multi-Agent Workflow System

This folder contains a multi-agent AI workflow system with specialized agents, skills, and chronological prompt tracking.

## 📁 Structure

```
.github/
├── agents/          # Custom agent definitions
│   ├── thinker.agent.md
│   ├── builder.agent.md
│   ├── tester.agent.md
│   └── hater.agent.md
├── skills/          # Reusable workflow skills
│   ├── save-prompt-chronologically/
│   ├── thinker-memory/
│   └── thinker-to-builder/
└── prompts/         # Chronologically numbered prompts
    ├── 001_xxx.md
    ├── 002_xxx.md
    └── thinker-memory.md
```

## 🤖 Agents

All agents may inspect the whole workspace when a task requires cross-folder context (skills, environment, integration points). Builder remains the implementation role; Thinker/Tester/Hater should still keep task artifacts organized under the relevant project folder (for PINN work: `battery_surrogate_agenticWorkflow_PINN/`).

### Thinker
**Role**: Architecture planning and task breakdown
- Plans system structure (cross-file and within-file)
- Maintains persistent memory
- Creates implementation checklists
- Hands off work to Builder

**Can**: Read files, search codebase, update memory, invoke subagents
**Cannot**: Create or edit files (that's Builder's job)

### Builder
**Role**: Code implementation
- Writes code and creates files
- Can rename and move files or folders across the workspace when required
- Implements features based on Thinker's plans
- Follows task checklists
- Reports completion status

**Can**: Create/edit files, run code, check errors
**Cannot**: None (full implementation access)

### Tester
**Role**: Quality assurance and testing
- Writes JUnit tests
- Runs test suites
- Finds bugs and validates functionality
- Reports issues to Builder

**Can**: Create test files, run tests, read code
**Cannot**: Edit production code (only test code)

### Hater
**Role**: Critical review and problem finding
- Reviews plans before implementation
- Finds architectural issues
- Challenges assumptions
- Provides constructive critique

**Can**: Read files, analyze code, document concerns
**Cannot**: Create or edit files (review only)

## 🔧 Skills

### save-prompt-chronologically
Saves prompts with automatic numbering (001, 002, 003...)
- **Used by**: All agents
- **Purpose**: Document decisions, plans, and important context
- **Format**: `{NNN}_{topic}.md`

### thinker-memory
Maintains Thinker's persistent memory with hybrid format
- **Used by**: Thinker agent only
- **File**: `.github/prompts/thinker-memory.md`
- **Format**: Summary at top + detailed chronological history

### thinker-to-builder
Creates implementation task checklists for Builder
- **Used by**: Thinker agent only
- **Output**: Numbered prompt with actionable checklist
- **Format**: Architecture overview + detailed steps

### modulus-env-smoke-test
Runs the verified Modulus environment smoke test using the correct environment
- **Used by**: Any agent
- **Purpose**: Confirm `modulus_env` is active and `testerOFMod.py` works
- **Checks**: Python path, Torch version, PINN smoke test, Modulus import

### modulus-agent-workflow
Describes the standard agent workflow for Modulus and PINN tasks in this repo
- **Used by**: Any agent
- **Purpose**: Keep Modulus work on the correct venv, smoke test, and CPU POC pattern
- **Checks**: Environment choice, smoke test, Modulus PINN workflow, local-vs-cloud split

## 🔄 Workflow

### Typical Flow

```
1. User requests feature/task
   ↓
2. THINKER analyzes and plans
   - Reads memory for context
   - Analyzes architecture
   - Updates memory with plan
   ↓
3. HATER reviews plan (optional but recommended)
   - Finds potential issues
   - Suggests improvements
   ↓
4. THINKER addresses concerns
   - Updates plan based on feedback
   - Creates Builder checklist
   ↓
5. BUILDER implements
   - Follows checklist systematically
   - Creates/edits files
   - Reports completion
   ↓
6. TESTER validates
   - Writes JUnit tests
   - Runs test suite
   - Reports bugs if found
   ↓
7. If bugs: BUILDER fixes → TESTER re-validates
   If clean: Done! ✅
```

### Invoking Agents

In VS Code with GitHub Copilot:
- Type `@thinker` to invoke Thinker
- Type `@builder` to invoke Builder
- Type `@tester` to invoke Tester
- Type `@hater` to invoke Hater

Or use subagent invocation:
```
Thinker: "Builder, implement the tasks in prompt 042"
Builder: "Tester, validate the implementation"
```

## 📝 Prompt Numbering System

All prompts are saved chronologically with zero-padded numbers:
- `001_initial-setup.md` - First prompt
- `002_thinker-plan.md` - Thinker's plan
- `003_builder-implementation.md` - Builder's checklist
- `004_hater-review.md` - Hater's critique
- etc.

**Benefits**:
- Easy chronological tracking
- Simple cross-references ("See prompt 042")
- Preserves conversation history
- Searchable archive

## 🎯 Tool Restrictions

Each agent has specific tool access to enforce role separation:

| Agent   | Read | Create/Edit Files | Run Commands | Invoke Subagents |
|---------|------|-------------------|--------------|------------------|
| Thinker | ✅   | ❌ | ✅ (read-only) | ✅ |
| Builder | ✅   | ✅ | ✅ | ✅ |
| Tester  | ✅   | ✅ (test files only) | ✅ | ✅ |
| Hater   | ✅   | ❌ | ✅ (read-only) | ❌ |

All agents can:
- Save prompts chronologically
- Edit their own agent setup
- Access memory and documentation

## 🚀 Getting Started

### First Time Setup
The system is already initialized! The structure is:
- ✅ Agents defined in `.github/agents/`
- ✅ Skills defined in `.github/skills/`
- ✅ Prompts folder ready at `.github/prompts/`
- ✅ Thinker memory initialized

### First Task
1. Invoke Thinker: `@thinker Plan the architecture for [your task]`
2. Thinker will analyze, plan, and hand off to Builder
3. Builder will implement
4. Tester will validate

### Example Usage
```
User: @thinker I need to add a new validation layer for user inputs

Thinker:
1. Reads thinker-memory.md
2. Analyzes existing code structure
3. Plans architecture (files, classes, integration)
4. Invokes @hater for review
5. Updates memory and creates prompt 001_builder-validation-layer.md
6. Invokes Builder: "Implement tasks in prompt 001"

Builder:
1. Reads prompt 001
2. Creates validation.py
3. Implements ValidationLayer class
4. Integrates with existing code
5. Reports completion

Tester:
1. Writes test_validation.py
2. Runs JUnit tests
3. Reports results
```

## 📚 File Descriptions

- **thinker.agent.md**: Thinker agent definition and instructions
- **builder.agent.md**: Builder agent definition and instructions
- **tester.agent.md**: Tester agent definition and instructions
- **hater.agent.md**: Hater agent definition and instructions
- **save-prompt-chronologically/SKILL.md**: Prompt numbering skill
- **thinker-memory/SKILL.md**: Memory management skill
- **thinker-to-builder/SKILL.md**: Handoff instruction skill
- **modulus-env-smoke-test/SKILL.md**: Verified Modulus environment smoke-test workflow
- **modulus-agent-workflow/SKILL.md**: Agent workflow for Modulus and PINN work in this repo
- **prompts/thinker-memory.md**: Thinker's persistent memory
- **prompts/{NNN}_*.md**: Chronological prompt archive

## 💡 Tips

1. **Always start with Thinker** for complex tasks - let it plan first
2. **Use Hater reviews** to catch issues early
3. **Check thinker-memory.md** to see project context and decisions
4. **Reference prompts by number** for easy cross-linking
5. **Let Builder focus on implementation** - don't skip the planning phase
6. **Tester validates everything** - don't assume it works

## 🔍 Troubleshooting

**Agent not found?**
- Ensure you're using `@agentname` syntax
- Check that agent files exist in `.github/agents/`

**Skill not working?**
- Skills are auto-discovered by description
- Check YAML frontmatter syntax

**Prompts not saving?**
- Ensure `.github/prompts/` folder exists
- Check permissions

## 📖 Learn More

- VS Code Agent Customization: See `.instructions.md` and `.agent.md` docs
- YAML frontmatter reference: Check agent-customization skill
- Subagent workflows: Read individual agent files

---

**System Status**: ✅ Initialized and ready for use
**Next Step**: Invoke `@thinker` with your first task!
