# Phase-Based Analysis API

This module provides structured, phase-based analysis endpoints for the AI Business Analyst Assistant.

## Overview

The new API follows a structured approach where each phase produces specific outputs that can be used by subsequent phases. The phases are:

1. **Phase 1: Problem Definition** - Analyze and structure the problem
2. **Phase 2: Requirements Analysis** - Feature analysis and user journeys
3. **Phase 3: Market Analysis** - Market research and competitive analysis
4. **Phase 7: Documentation** - Generate various document types

## API Endpoints

Base URL: `/api/v2/phases`

### Phase 1: Problem Definition

**Endpoint:** `POST /users/{user_id}/phase1/problem-definition`

**Input (required fields marked with *):**
```json
{
    "problem_description": "* Describe the problem in your own words",
    "target_users": "* Who experiences this problem?",
    "why_it_matters": "* What's the impact?",
    "pain_points": ["Optional list of specific pain points"],
    "has_existing_solutions": false,
    "current_solutions": "Optional: what solutions exist and their issues"
}
```

**Output:**
1. **Quality Score** - Overall 0-100% assessment with dimensions:
   - Problem Clarity
   - Pain Point Validity
   - Market Readiness
   - User Specificity
   - Scope Definition

2. **Normalized Problem Summary** - Solution-agnostic rewrite (max 120 words)

3. **User Persona Clarification** - Primary and secondary personas

4. **Concrete Pain Moments** - 3-6 vivid scenes (not abstract labels)

5. **Root Cause Analysis** - Categorized root causes

6. **Existing Solutions Analysis** - Why current solutions fail

7. **Impact & Stakes** - Quantified or bounded impacts

8. **Scope Boundary** - Explicit exclusions

9. **Transition Questions** - 5-7 questions for Phase 2

---

### Phase 2: Requirements Analysis

#### Feature Analyzer

**Endpoint:** `POST /users/{user_id}/phase2/feature-analyzer`

**Input:**
```json
{
    "desired_features": ["* List of features"],
    "mvp_goal": "* SMART MVP goal (e.g., 'Help students submit polished essay within 48 hours')",
    "deadline": "Optional: '3 months'",
    "team_skill_level": "Optional: solo-developer|junior-team|senior-team|mixed-experience",
    "additional_constraints": "Optional constraints"
}
```

**Output:**
1. **Normalized Feature List** - Deduplicated, renamed in user-outcome language, categorized (core/enhancement/nice-to-have)

2. **Functional Requirements** - Categorized (AI/ML, Auth, DB, UI, etc.), MoSCoW priority, complexity

3. **Non-Functional Requirements** - Similar categorization

4. **MVP Scope Summary** - Minimum viable features for launch

5. **Scope Warnings** - Potential risks and concerns

#### User Journey Generator

**Endpoint:** `POST /users/{user_id}/phase2/user-journey`

**Input:**
```json
{
    "selected_feature": "* Feature name from the list"
}
```

**Output:**
- Step-by-step journey with:
  - Goal per step
  - User actions
  - System responses
  - Success criteria
  - Potential issues

---

### Phase 3: Market Analysis

**Endpoint:** `POST /users/{user_id}/phase3/market-analysis`

**Input (all optional):**
```json
{
    "geographic_scope": "local|regional|multi-regional|national|international|global",
    "industry_context": "e.g., 'EdTech'",
    "competitors": ["Known competitor names"]
}
```

**Output:**
1. **Market Research Summary** - With real website links for statistics

2. **Porter's Five Forces Analysis** - Full framework analysis

3. **Competitor Analysis** - For each competitor:
   - Business Model
   - Strengths/Weaknesses
   - Target Customer
   - Opportunities/Threats

4. **Generated USPs** - Unique selling points with positioning statement

---

### Phase 7: Documentation

**Endpoint:** `POST /users/{user_id}/phase7/documentation`

**Input:**
```json
{
    "document_type": "* academic-report|software-engineering|business-proposal|startup-pitch",
    "project_title": "Optional title",
    "author_name": "Optional author/team name",
    "additional_context": "Optional specific requirements"
}
```

**Output varies by document type:**

- **Academic Report** - Follows VinUni CECS Capstone template
- **Software Engineering (SRS)** - IEEE 830 format
- **Business Proposal** - Full business proposal structure
- **Startup Pitch** - 10-12 slide pitch deck

---

### Session Management

**Get Session Summary:**
`GET /users/{user_id}/session`

**Clear Session:**
`DELETE /users/{user_id}/session`

## Usage Flow

1. Start with Phase 1 to define and analyze the problem
2. Use Phase 2 to analyze features (automatically uses Phase 1 context)
3. Use Phase 3 for market analysis (uses Phase 1 & 2 context)
4. Generate documentation with Phase 7 (uses all previous phases)

## Templates

The system supports two document templates:
- `back_end/templates/VinUni-CECS-Capstone-Project-template.docx` - Academic reports
- `back_end/templates/srs_template-ieee.docx` - IEEE SRS documents

## Prompts

All prompts are stored in `common/prompts.py` for easy modification:

### Phase 1 Prompts:
- `PHASE1_QUALITY_SCORING_PROMPT`
- `PHASE1_NORMALIZE_PROBLEM_PROMPT`
- `PHASE1_USER_PERSONA_PROMPT`
- `PHASE1_PAIN_MOMENTS_PROMPT`
- `PHASE1_ROOT_CAUSE_PROMPT`
- `PHASE1_EXISTING_SOLUTIONS_PROMPT`
- `PHASE1_IMPACT_STAKES_PROMPT`
- `PHASE1_SCOPE_BOUNDARY_PROMPT`
- `PHASE1_TRANSITION_QUESTIONS_PROMPT`

### Phase 2 Prompts:
- `PHASE2_NORMALIZE_FEATURES_PROMPT`
- `PHASE2_FUNCTIONAL_REQUIREMENTS_PROMPT`
- `PHASE2_NON_FUNCTIONAL_REQUIREMENTS_PROMPT`
- `PHASE2_MVP_SCOPE_PROMPT`
- `PHASE2_SCOPE_WARNINGS_PROMPT`
- `PHASE2_USER_JOURNEY_PROMPT`

### Phase 3 Prompts:
- `PHASE3_MARKET_RESEARCH_PROMPT`
- `PHASE3_PORTERS_FIVE_FORCES_PROMPT`
- `PHASE3_COMPETITOR_ANALYSIS_PROMPT`
- `PHASE3_USP_GENERATION_PROMPT`

### Phase 7 Prompts:
- `PHASE7_ACADEMIC_REPORT_PROMPT`
- `PHASE7_SRS_DOCUMENT_PROMPT`
- `PHASE7_BUSINESS_PROPOSAL_PROMPT`
- `PHASE7_STARTUP_PITCH_PROMPT`
