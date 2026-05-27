from transformers import pipeline

MODEL_NAME = "dslim/bert-base-NER"

def normalize_label(label: str) -> str:
    label = label.upper()
    if "PER" in label:
        return "Person"
    elif "ORG" in label:
        return "Organization"
    elif "LOC" in label:
        return "Location"
    elif "DATE" in label:
        return "Date"
    elif "MISC" in label:
        return "Miscellaneous"
    return "Other"

ner_pipeline = pipeline(
    "ner",
    model=MODEL_NAME,
    aggregation_strategy="simple"
)

def get_named_entities(text: str):
    if not text.strip():
        return []

    results = ner_pipeline(text)

    return [
        {
            "entity": item.get("word", ""),
            "label": normalize_label(item.get("entity_group", item.get("entity", ""))),
            "score": round(float(item.get("score", 0)), 4),
            "start": item.get("start"),
            "end": item.get("end")
        }
        for item in results
    ]