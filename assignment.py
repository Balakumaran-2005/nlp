def analyze_text_corpus(corpus):
  """
  Analyzes a text corpus and produces unigrams, bigrams, trigrams, bigram probabilities,
  and next word prediction functionality.

  Args:
      corpus (str): The text corpus to analyze.

  Returns:
      dict: A dictionary containing the analysis results.
  """

  # Preprocess the corpus: lowercase, split into words, remove punctuation
  words = [word.lower() for word in corpus.split() if word.isalpha()]

  # 1. Unigrams: Count occurrences of each word
  unigrams = {}
  for word in words:
    unigrams[word] = unigrams.get(word, 0) + 1

  # 2. Bigrams: Count occurrences of pairs of words
  bigrams = {}
  for i in range(len(words) - 1):
    first_word = words[i]
    second_word = words[i + 1]
    bigram = (first_word, second_word)
    bigrams[bigram] = bigrams.get(bigram, 0) + 1

  # 3. Trigrams: Count occurrences of triples of words
  trigrams = {}
  for i in range(len(words) - 2):
    first_word = words[i]
    second_word = words[i + 1]
    third_word = words[i + 2]
    trigram = (first_word, second_word, third_word)
    trigrams[trigram] = trigrams.get(trigram, 0) + 1

  # 4. Bigram Probabilities: Calculate probability of second word given first word (with Laplace smoothing)
  bigram_probs = {}
  total_words = len(words) - 1  # Avoid double-counting for bigrams
  for bigram, count in bigrams.items():
    first_word, second_word = bigram
    first_word_count = unigrams.get(first_word, 0)
    bigram_probs[bigram] = (count + 1) / (first_word_count + len(unigrams))  # Laplace smoothing

  # 5. Next Word Prediction: Predict the most likely next word given a bigram
  def next_word_prediction(bigram):
    if bigram not in bigram_probs:
      return None  # Handle unseen bigrams (e.g., with backoff)

    next_word_candidates = [(word, prob) for word, prob in bigram_probs.items() if word[0] == bigram[1]]

    # Return the word with the highest probability (consider improvements like beam search for more complex predictions)
    return max(next_word_candidates, key=lambda x: x[1])[0]

  return {
      "unigrams": unigrams,
      "bigrams": bigrams,
      "trigrams": trigrams,
      "bigram_probs": bigram_probs,
      "next_word_prediction": next_word_prediction,
  }

# Example usage
corpus = "This is a sample text corpus to analyze for n-grams and next word prediction."
results = analyze_text_corpus(corpus)

print("1. Unigrams:")
for word, count in results["unigrams"].items():
  print(f"{word}: {count}")

print("\n2. Bigrams:")
for bigram, count in results["bigrams"].items():
  print(f"{bigram}: {count}")

print("\n3. Trigrams:")
for trigram, count in results["trigrams"].items():
  print(f"{trigram}: {count}")

print("\n4. Bigram Probabilities:")
for bigram, prob in results["bigram_probs"].items():
  print(f"{bigram}: {prob:.4f}")  # Format probability with 4 decimal places

# Example next word prediction
bigram = ("this", "is")
next_word = results["next_word_prediction"](bigram)
print(f"\n5. Next word prediction for '{bigram}': {next_word}")
