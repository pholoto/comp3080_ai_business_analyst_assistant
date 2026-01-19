from AI.features import FeatureContext, build_default_registry
from AI.llm import get_default_client
from AI.memory import SessionManager

# 1) Create a session (stores conversation state)
session = SessionManager().create_session()

# 2) Get an LLM client and feature registry
llm = get_default_client()
registry = build_default_registry()

# 3) Choose a feature and your user message
feature_key = "solution_design"  # pick one from the list above
message = "Propose a high-level architecture for our mobile app."

# 4) Create the feature and run it
context = FeatureContext(session=session, llm=llm)
feature = registry.create(feature_key, context)
result = feature.run(message, context=context)

# 5) Show the output (what you render in front-end)
print("Title:", result.title)
print("Summary:\n", result.summary)
print("Data:")
print(result.data)  # dict with structured fields (e.g., components, steps, matrices)