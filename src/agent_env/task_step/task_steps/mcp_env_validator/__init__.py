TOOL_CORRECTNESS_OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "error": {"type": "string"},
                        "justification": {"type": "string"},
                    },
                    "required": ["tool_name", "passed", "error", "justification"],
                },
            },
        },
        "required": ["results"],
    },
}

TOOL_CORRECTNESS_PROMPT = """You have access to MCP tools. Your job is to validate that each tool works correctly.

Assume that some data may already exist in the system.

Follow this order:

1. **Create test data first**: Use write/create tools to build up test data. Some writes may depend on other writes (e.g. creating a record may require first creating a table, posting a message may require first having a channel). Figure out the dependency order and create prerequisite data before testing dependent tools.

2. **Read-back verification**: After creating data, use read/get/list tools to verify the data was persisted and matches what was written.

3. **Update verification**: If update tools exist, modify an entity you created and verify the changes are reflected when reading it back.

4. **Delete verification**: If delete tools exist, delete an entity and verify it is no longer returned by read/list tools.

5. **Search verification**: If search tools exist, verify they can find data you created.

A tool passes if it behaves correctly — including returning a well-structured error for invalid inputs. A tool fails only if it has an actual defect: crashes, returns malformed data, silently drops writes, or behaves inconsistently with its description.

If a tool cannot be tested because there is no way to create its prerequisites through the available toolset (e.g. no create tool exists for a required parent entity), mark it as passed with a justification noting the limitation.

Report your findings as structured JSON. For each tool, set passed=true if it works correctly, or passed=false with an error description if it fails."""
