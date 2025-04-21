

### **MCP (Multi-Component Protocol)**

- MCP allows an AI agent to take actions on your behalf.
- It defines an open-ended protocol where the AI is empowered to autonomously decide and execute actions for you.

---

### **Components: Host, Client, Server, Tools**

- **Host**: The AI interface (e.g., LLMs) responsible for interpreting and planning tasks.
- **Client**: The bridge that connects the Host to one or more Servers.
- **Server**: The backend system that exposes your tools or capabilities.
- **Tools**: Functions or APIs that the AI can call to perform specific tasks.

---

### **Example Scenario**  
**Task**: *"Prepare coffee and check my free schedule."*

- **Host**: The AI determines that two actions are needed — `makeCoffee` and `checkSchedule`.
- **Client**: Sends a request to the MCP server on behalf of the Host.
- **Server**: Executes the functions `make_coffee()` and `check_schedule()`.
- **Tools**: The actual functions that order the coffee and return your available schedule.

---

### **Comparison: API vs RAG vs Tool Use vs MCP**

#### **Tool Usage Levels**

| Method        | Description                             |
|---------------|-----------------------------------------|
| **API**       | Manual invocation of a function         |
| **RAG**       | Retrieval-based, no action taken        |
| **Tool Calling** | AI calls a function via wrappers     |
| **MCP**       | Fully autonomous tool calling by the AI |

#### **Tool Discovery Capabilities**

| Method  | Tool Discovery |
|---------|----------------|
| API     | ❌ No           |
| RAG     | ❌ No           |
| MCP     | ✅ Fully Automatic |
