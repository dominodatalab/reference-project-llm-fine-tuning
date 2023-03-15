from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import pipeline

model = BertForSequenceClassification.from_pretrained("finbert-sentiment",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("finbert/tokenizer")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def predict_sentiment(sentence):
    results = nlp(sentence)
    return results[0]

def main():
    test_sentence = "there is a shortage of capital, and we need extra financing"
    sentiment = predict_sentiment(test_sentence)
    print("sentence   : {}".format(test_sentence))
    print("prediction : {}".format(sentiment["label"]))
    print("score      : {}".format(sentiment["score"]))

if __name__ == "__main__":
    main()
