# Project Requirements

## Overview

This tool is designed to analyze options trading data, specifically focusing on PUT options with negative quantities (sold puts), compute remaining income potential, and detect spread strategies to assess risk-reward profiles. This tool operates via a Command-Line Interface (CLI) with a textual user interface (TUI) feel and supports in-depth filtering, calculations, and sorting of data relevant to equity options.

The input will be in a TSV format.

---

## Project Management Plan

### Development Phases

1. **Planning**

   - Review requirements and draft data models
   - Setup GitHub repository
   - Define folder structure

2. **Core Functionality**

   - File parser and validator
   - Filtering logic for sold puts
   - Calculations engine
   - Sorting and display logic

3. **Spread Detection Engine**

   - Pairing logic for spreads
   - Calculation of spread metrics
   - Integration into main display pipeline

4. **TUI Enhancement**

   - Implement interactive filtering/sorting
   - Highlight spreads visually

5. **Testing & Validation**

   - Unit tests for calculation modules
   - Mock data file for regression

### Task Tracking Table

Maintain a `tasks.md` file. Suggested columns:

| Task ID | Description           | Assigned To | Status      | Priority | Notes                  |
| ------- | --------------------- | ----------- | ----------- | -------- | ---------------------- |
| T001    | Implement file parser | Dev A       | In Progress | High     | Use pandas             |
| T002    | Compute income %/day  | Dev B       | Not Started | Medium   |                        |
| T003    | Build CLI menu        | Dev C       | Done        | Medium   | Use argparse           |
| T004    | Spread pairing logic  | Dev A       | Not Started | High     | Match by ticker/expiry |

# Agentic SOP

- When given a new task, add it to the task tracking table
- When working on a task, update its status
- When completing a task, mark it as complete
- When you are given ambiguities, ask the user to resolve them and suggest an industry-standard best-practice recommendation.
- Use modular code practices. Refactor when necessary
