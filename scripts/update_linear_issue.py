#!/usr/bin/env python3
import os
import requests

# Get API key from environment
API_KEY = os.environ.get('LINEAR_API_KEY')
if not API_KEY:
    print("Please set LINEAR_API_KEY environment variable")
    exit(1)

# Linear GraphQL endpoint
GRAPHQL_URL = "https://api.linear.app/graphql"

# Headers
headers = {
    "Authorization": API_KEY,
    "Content-Type": "application/json"
}

# First, let's find the issue DEV-800
search_query = """
query FindIssue {
  issues(filter: { identifier: { eq: "DEV-800" } }) {
    nodes {
      id
      identifier
      title
      description
    }
  }
}
"""

response = requests.post(GRAPHQL_URL, json={"query": search_query}, headers=headers)
data = response.json()

if not data.get('data', {}).get('issues', {}).get('nodes'):
    print("Issue DEV-800 not found")
    exit(1)

issue = data['data']['issues']['nodes'][0]
issue_id = issue['id']
print(f"Found issue: {issue['identifier']} - {issue['title']}")

# Add a comment with our implementation plan
comment_mutation = """
mutation AddComment($issueId: String!, $body: String!) {
  commentCreate(input: { issueId: $issueId, body: $body }) {
    success
    comment {
      id
    }
  }
}
"""

comment_body = """Cursor via MCP:

## Implementation Plan for LoCoMo Evaluation on Honcho

Based on analysis of the codebase and requirements, here's the implementation plan:

### Architecture Overview

1. **Data Model Mapping**:
   - LoCoMo samples (10 conversations) → Honcho Apps
   - LoCoMo speakers (2 per conversation) → Honcho Users
   - LoCoMo sessions → Honcho Sessions (preserving temporal boundaries)
   - LoCoMo dialogs → Honcho Messages

2. **Cross-Session Retrieval**:
   - Leverage Honcho's built-in dialectic API for cross-session information retrieval
   - The dialectic API automatically stores facts in user collections and retrieves relevant information across all sessions
   - No need to force all data into a single session

3. **Message Duplication Strategy**:
   - Each message stored twice to maintain both speaker perspectives
   - Speaker A's messages: stored with A as "user" and B's responses as "assistant"
   - Speaker B's messages: stored with B as "user" and A's responses as "assistant"

### Implementation Components

1. **Data Ingestion Script** (`scripts/ingest_locomo_to_honcho.py`):
   - Parse LoCoMo dataset
   - Create apps for each conversation
   - Create users for each speaker
   - Create sessions with timestamps
   - Store messages with proper metadata (date, original session, dialog ID)

2. **Honcho Utils Module** (`task_eval/honcho_utils.py`):
   - Implement `get_honcho_answers()` function
   - Handle user/session identification based on questions
   - Interface with Honcho's dialectic API
   - Format responses for evaluation compatibility

3. **Evaluation Script** (`scripts/evaluate_honcho.sh`):
   - Similar structure to existing evaluation scripts
   - Call modified evaluate_qa.py with Honcho model type

4. **Modified Evaluation** (`evaluate_qa.py`):
   - Add elif clause for Honcho
   - Import and use honcho_utils module

### Key Findings

- Evidence tracking is not required for answer correctness (only used for optional recall metric)
- Answer evaluation is based on F1/text matching, not on providing correct dialog IDs
- Honcho's dialectic API handles cross-session retrieval automatically through its vector store

### Next Steps

Starting implementation of the data ingestion script to parse LoCoMo dataset and populate Honcho instance.
"""

variables = {
    "issueId": issue_id,
    "body": comment_body
}

response = requests.post(GRAPHQL_URL, json={"query": comment_mutation, "variables": variables}, headers=headers)
result = response.json()

if result.get('data', {}).get('commentCreate', {}).get('success'):
    print("Successfully added comment to issue DEV-800")
else:
    print("Failed to add comment:", result) 