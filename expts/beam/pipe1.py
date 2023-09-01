

# Template for each interface
def load_models():
    model_state = {}
    model = pipeline(
        "sentiment-analysis", model="siebert/sentiment-roberta-large-english"
    )

    return model

def run_inference(**inputs):
    # Retrieve the model from the loader
    model = inputs["context"]

    result = model(inputs["text"], truncation=True, top_k=2)
    prediction = {i["label"]: i["score"] for i in result}

    return {"prediction": prediction}

if __name__ == '__main__':
    model = load_models()
    inputs = {}
    inputs["context"] = model
    # Fill necessary inputs other than the context here

    #################################
    run_inference(input)