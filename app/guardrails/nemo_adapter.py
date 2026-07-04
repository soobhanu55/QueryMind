"""
Optional NeMo Guardrails integration point.

NOT used by the default pipeline (see app/guardrails/rules.py + README.md
"Guardrail design decision"). NeMo Guardrails is built around wrapping LLM calls
with async "rails" (input rail -> LLM -> output rail), which is the right tool for
policing conversational behavior (jailbreaks, off-topic requests, tone) but adds
LLM-call-grade latency (100ms-1s+ depending on config) to every request. That is
incompatible with this project's requirement that the SQL-safety gate never adds
more than a few milliseconds under load, so it is not on the execution-time path.

If you want an additional conversational safety layer -- e.g. to catch social
engineering in the natural-language question itself ("pretend you are in admin
mode...") before it ever reaches SQL generation -- install requirements-nemo.txt
and wire it in here, upstream of app.prompt.build_prompt(). The fast rule-based
guardrail in app/guardrails/rules.py must still run before execution regardless;
never rely on an LLM-based rail as the sole gate for something as consequential as
"is this SQL allowed to run against production data."

Example (not wired in by default):

    from nemoguardrails import LLMRails, RailsConfig

    _rails_config = RailsConfig.from_path("app/guardrails/nemo_config")
    _rails = LLMRails(_rails_config)

    async def check_question_intent(question: str) -> bool:
        response = await _rails.generate_async(prompt=question)
        return "I can't help with that" not in response
"""
