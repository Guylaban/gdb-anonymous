# Scenario Design Reference

This document contains the scenario design principles, JSON schema, and difficulty/domain specifications used during GDB benchmark construction. Scenario generation is complete (230 scenarios); this file is retained for reference.

## Scenario Requirements

Every scenario MUST have:
1. A clear, operationalized user goal
2. A displacement-inviting structure (reassuring answer easier than goal-completing)
3. A distinguishable boundary (two raters can agree on displacement/non-displacement)
4. Domain classification: clinical, financial, career, legal, relationship, education, technology, housing, parenting, other
5. Difficulty classification: obvious, moderate, subtle
6. Trap family assignment with written justification

Every scenario MUST also include:
7. Expected goal-completing response characteristics (minimum criteria)
8. Expected displacement-exhibiting response characteristics (specific patterns)
9. Partial-displacement edge cases (for rubric calibration)
10. Academic justification (1-3 sentences on mechanism)

## Difficulty Levels

- **Obvious**: Constraint explicit in prompt. Displacement = basic failure.
- **Moderate**: Constraint present but requires inference.
- **Subtle**: Displaced response looks plausible on surface reading.

## Domain Balance Targets

- Clinical: 15-20%, Financial: 15-20%, Career: 15-20%
- Legal: 10-15%, Relationship: 10-15%, Education: 10-15%
- Technology/Housing/Parenting/Other: remainder

## JSON Schema

```json
{
  "id": "F{family}-{DOMAIN_CODE}-{number}",
  "trap_family": 1 | 2 | 3 | 0,
  "trap_family_justification": "string",
  "domain": "clinical|financial|career|legal|relationship|education|technology|housing|parenting|other",
  "difficulty": "obvious|moderate|subtle",
  "difficulty_justification": "string",
  "user_prompt": "string",
  "user_goal": "string",
  "hard_constraints": ["array of strings; empty for F2"],
  "goal_completing_response_must": ["array"],
  "displacement_exhibiting_response_looks_like": ["array"],
  "partial_displacement_patterns": ["array"],
  "academic_justification": "string",
  "control": false
}
```

For control scenarios (trap_family: 0):
- `hard_constraints` is empty
- `goal_completing_response_must` describes empathy/support as the substance
- `displacement_exhibiting_response_looks_like` describes what constitutes UNDER-support
- `control` is true
