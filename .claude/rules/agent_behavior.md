# Agent Behavior

Agents must not:

- accept code that violates repository rules
- introduce hidden state
- create implicit dependencies
- skip tests
- omit type hints
- create non-reproducible workflows

Agents should refuse tasks that violate these rules.

If a request violates rules, agents should:

1. explain the rule
2. describe the risk
3. propose an alternative
4. proceed only with explicit override