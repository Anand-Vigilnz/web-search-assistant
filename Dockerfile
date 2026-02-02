# MCP Agent Chatbot - Streamlit app with LangChain
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY agent.py .
COPY mcp_tools.py .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Run the app; use 0.0.0.0 so it's accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
