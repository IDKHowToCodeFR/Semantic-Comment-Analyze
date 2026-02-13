from typing import Any

# Ponytail Ultra: Heuristic-based Domain Rules (No ML models)


def analyze_tone(intent: str, sentiment: dict[str, Any]) -> str:
    label = sentiment.get("label", "NEUTRAL")
    score = sentiment.get("score", 0.0)
    if label == "NEGATIVE (SARCASTIC)": return "Mocking"

    if intent == "Bug Report":
        return "Frustrated" if label == "NEGATIVE" else "Concerned"
    elif intent == "Feature Request":
        return "Excited" if label == "POSITIVE" else "Hopeful"
    elif intent == "Complaint":
        return "Angry" if label == "NEGATIVE" and score > 0.8 else "Frustrated"
    elif intent == "Praise":
        return "Appreciative"
    elif intent == "Question":
        return "Confused" if label == "NEGATIVE" else "Curious"
    return "Neutral"


def calculate_urgency(intent: str, sentiment: dict[str, Any]) -> str:
    label = sentiment.get("label", "NEUTRAL")
    score = sentiment.get("score", 0.0)
    if label == "NEGATIVE (SARCASTIC)": return "High"

    if intent == "Bug Report":
        return "High" if label == "NEGATIVE" else "Medium"
    elif intent == "Complaint":
        return "High" if score > 0.8 and label == "NEGATIVE" else "Medium"
    elif intent == "Feature Request":
        return "Low"
    return "Low"


def recommend_action(intent: str, urgency: str, sentiment: dict[str, Any]) -> str:
    if sentiment and sentiment.get("label") == "NEGATIVE (SARCASTIC)": return "Damage Control"
    if intent == "Bug Report":
        return "Log Jira Ticket" if urgency == "High" else "Monitor Issue"
    elif intent == "Feature Request":
        return "Add to Roadmap"
    elif intent == "Complaint":
        return "Route to Support"
    elif intent == "Question":
        return "Update Docs/FAQ"
    elif intent == "Praise":
        return "Share with Team"
    return "No Action Needed"


def evaluate_business_context(intent: str, sentiment: dict[str, Any]) -> dict[str, str]:
    """Orchestrates business rules and returns a consolidated context."""
    tone = analyze_tone(intent, sentiment)
    urgency = calculate_urgency(intent, sentiment)
    action = recommend_action(intent, urgency, sentiment)
    return {
        "tone": tone,
        "urgency": urgency,
        "recommended_action": action
    }
