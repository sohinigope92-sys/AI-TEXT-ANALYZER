from transformers import pipeline

# Load models
sentiment_analyzer = pipeline("sentiment-analysis")
text_generator = pipeline("text-generation", model="gpt2")

def analyze_text():
    text = input("Enter text: ")

    # Sentiment Analysis
    sentiment = sentiment_analyzer(text)
    print("\nSentiment Analysis Result:")
    print(sentiment)

    # Text Generation
    generated = text_generator(text, max_length=30, num_return_sequences=1)
    print("\nGenerated Text:")
    print(generated[0]['generated_text'])


if __name__ == "__main__":
    analyze_text()
