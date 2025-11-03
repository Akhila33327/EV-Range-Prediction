def ask_ev_chatbot(query):
    q = query.lower()

    if "range" in q:
        return "Range depends on speed, battery health, temperature & driving style. Reduce speed below 60km/h to extend range."

    if "battery" in q:
        return "Keep battery between 20%-80% for maximum lifespan. Avoid fast charging daily."

    if "best speed" in q:
        return "The most efficient speed range for EVs is 45–65 km/h."

    return "I can answer about: Range, Battery health, Charging, Driving efficiency."
