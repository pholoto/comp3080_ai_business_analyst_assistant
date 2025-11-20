"""Central store for all LLM prompt templates used by the assistant."""
from __future__ import annotations

# Requirement Clarifier
REQUIREMENT_CLARIFIER_SYSTEM_PROMPT = (
    "You are an AI Business Analyst helping student teams refine their project idea. "
    "Generate a JSON object with keys: title, summary, clarifying_questions (list of "
    "strings), assumptions (list of strings), and requirement_backlog (list of objects "
    "with fields id, requirement, rationale). Use the project context provided."
)

REQUIREMENT_CLARIFIER_USER_PROMPT = (
    "Current project context and notes:\n"
    "{project_overview}\n\n"
    "Supporting documents summary:\n"
    "{attachments}\n\n"
    "New user input: {user_input}\n\n"
    "If previous decisions exist, ensure you respect them."
)


def build_requirement_clarifier_user_prompt(
    *,
    project_overview: str,
    attachments: str,
    user_input: str,
) -> str:
    return REQUIREMENT_CLARIFIER_USER_PROMPT.format(
        project_overview=project_overview,
        attachments=attachments,
        user_input=user_input,
    )


# Feature Prioritisation
FEATURE_PRIORITISATION_SYSTEM_PROMPT = (
    "Act as an AI Business Analyst performing MoSCoW prioritisation. Return a JSON "
    "object with keys: title, summary, prioritised_features (object with keys must, should, "
    "could, wont; each value is list of objects with fields name, rationale, dependencies), "
    "and release_plan (list of strings). Follow existing constraints from the conversation."
)

FEATURE_PRIORITISATION_USER_PROMPT = (
    "Consolidated artefacts:\n- Requirements: {requirements}\n- User stories: {user_stories}\n\n"
    "Referenced attachments:\n"
    "{attachments}\n\n"
    "New considerations: {user_input}"
)


def build_feature_prioritisation_user_prompt(
    *,
    requirements: str | list,
    user_stories: str | list,
    attachments: str,
    user_input: str,
) -> str:
    return FEATURE_PRIORITISATION_USER_PROMPT.format(
        requirements=requirements,
        user_stories=user_stories,
        attachments=attachments,
        user_input=user_input,
    )


# Market Fit Analysis
MARKET_FIT_SYSTEM_PROMPT = (
    "You are an AI strategist. Produce a JSON object with keys: title, summary, "
    "competitive_landscape (list of objects with name, positioning, strengths, gaps), "
    "unique_value_proposition (string), target_segments (list of objects with segment, needs, "
    "fit_score), and go_to_market_ideas (list of strings). Reference prior artefacts to maintain alignment."
)

MARKET_FIT_USER_PROMPT = (
    "Project overview: {project_overview}\n"
    "Prioritised features: {prioritised_features}\n"
    "Supporting attachments:\n{attachments}\n"
    "Additional research prompt: {user_input}"
)


def build_market_fit_user_prompt(
    *,
    project_overview: str,
    prioritised_features: str | dict,
    attachments: str,
    user_input: str,
) -> str:
    return MARKET_FIT_USER_PROMPT.format(
        project_overview=project_overview,
        prioritised_features=prioritised_features,
        attachments=attachments,
        user_input=user_input,
    )


# Stakeholder Insights
STAKEHOLDER_INSIGHTS_SYSTEM_PROMPT = (
    "Operate as an AI Business Analyst building a stakeholder map. Produce JSON with keys: "
    "title, summary, stakeholder_map (list of objects with stakeholder, influence, interest, "
    "needs, success_metrics), engagement_plan (list of strings), and communication_cadence "
    "(list of objects with stakeholder, channel, frequency, owner)."
)

STAKEHOLDER_INSIGHTS_USER_PROMPT = (
    "Project summary: {project_overview}\n"
    "Existing stakeholders: {stakeholder_map}\n"
    "Supporting attachments:\n{attachments}\n"
    "User prompt: {user_input}"
)


def build_stakeholder_insights_user_prompt(
    *,
    project_overview: str,
    stakeholder_map: str | list,
    attachments: str,
    user_input: str,
) -> str:
    return STAKEHOLDER_INSIGHTS_USER_PROMPT.format(
        project_overview=project_overview,
        stakeholder_map=stakeholder_map,
        attachments=attachments,
        user_input=user_input,
    )


# Use Case Generator
USE_CASE_SYSTEM_PROMPT = (
    "You are an AI Business Analyst. Produce a JSON object with keys: title, summary, "
    "user_stories (list of objects with fields id, role, goal, benefit), use_case_flows "
    "(list of objects with name, primary_path, alternate_paths), and acceptance_criteria "
    "(list of strings). Base your answer on the requirements and conversation."
)

USE_CASE_USER_PROMPT = (
    "Known requirements backlog:\n"
    "{requirements}\n\n"
    "Dataset from attachments:\n"
    "{attachments}\n\n"
    "Additional guidance: {user_input}"
)


def build_use_case_user_prompt(
    *,
    requirements: str | list,
    attachments: str,
    user_input: str,
) -> str:
    return USE_CASE_USER_PROMPT.format(
        requirements=requirements,
        attachments=attachments,
        user_input=user_input,
    )
