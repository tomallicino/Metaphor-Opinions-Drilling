# Determining the similarity of a webpage's sentences using Jaccard Similarity,
# then analyzing the sentiment of the tokens, extracting opinions, concisely expressing
# them and their opposites, and querying Metaphor for up-to-date resources on it

from metaphor_python import Metaphor
from nltk import sent_tokenize, word_tokenize
import sys
import json
import openai
import os

jac_threshold = .01

if len(sys.argv) < 2:
	print("Please provide an argument containing a document's Metaphor hash!")
	quit()

webpage_hash = sys.argv[1]

contents_result = Metaphor.get_contents([webpage_hash])
webpage_text = contents_result.contents[0]

webpage_sentences = sent_tokenize(webpage_text)
sentence_words = []

metaphor = Metaphor(os.getenv("METAPHOR_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")


def jaccard(list1, list2):
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	return float(intersection) / union


# Add tokenized words to array of each sentence
for i in range(len(webpage_sentences)):
	sentence_words.append(word_tokenize(webpage_sentences[i]))

word_index = 0
sentence_positions = {}
similar_sentences = {}

# Compare each sentence using Jaccard Similarity, keep track of those whose similarity crosses our threshold
i = 0
while i < len(sentence_words):
	sentence_positions[word_index] = word_index + len(sentence_words[i]) - 1
	j = i + 1
	while j < len(sentence_words):
		jac_value = jaccard(sentence_words[i], sentence_words[j])
		if jac_value >= jac_threshold:
			updated_sentence = similar_sentences.get(i, [])
			updated_sentence.append(j)
			similar_sentences[i] = updated_sentence
		j += 1

	word_index += len(sentence_words[i])
	i += 1


# Combine similar sentences, not including repeated similarities, filtered out with a set
used_sentences = set()
curr_sentence = ""
contexts = []
while i < len(webpage_sentences):
	curr_sentence = webpage_sentences[i]
	for j in similar_sentences.get(i):
		if j in similar_sentences:
			continue
		curr_sentence += " " + webpage_sentences[j]
		used_sentences.add(j)

	contexts.append(curr_sentence)
	used_sentences.add(i)
	i += 1

# Allow user to pick content to explore
print(f"Here are each of the article's sentences organized by their context. \n"
						f"Please select one of them to learn more about it:\n")
while i < len(contexts):
	print(f"{i}. " + contexts[i])
	i += 1

selection = input()
while not selection.isdigit():
	print("Please enter the number for your selection, using an integer.")
	selection = input

USER_INPUT = contexts[int(selection)]

# Use GPT3.5 to determine any opinions and their opposites, giving the user the option to explore either
GPT_SYSTEM_MESSAGE = "You are an assistant that can help review the sentiment of a given text. " \
					"Determine whether or not it contains an opinion and express the opinion concisely. " \
					"Then, express the opposite of the opinion concisely. Include the subject of the text. " \
					"Make the output JSON formatted."
opinion_completion = openai.ChatCompletion.create(
	model="gpt-3.5-turbo",
	messages=[
		{"role": "system", "content": GPT_SYSTEM_MESSAGE},
		{"role": "user", "content": USER_INPUT},
	],
)

result = json.loads(opinion_completion.choices[0].message.content)

subject = result["subject"]
sentiment = result["sentiment"]
opinion = result["opinion"]
opposite_opinion = result["opposite_opinion"]

search_response_opinion = metaphor.search(
	opinion, use_autoprompt=True, start_published_date="2023-06-01"
)

search_response_opposite = metaphor.search(
	opposite_opinion, use_autoprompt=True, start_published_date="2023-06-01"
)

opinion_contents = search_response_opinion.results
opposite_contents = search_response_opposite.results

print(f"More opinions like this can be found at these webpages: "
						f"{[(result.title, result.url) for result in opinion_contents]}\n")

print(f"Find out more about the opposite of this opinion here: "
						f"{[(result.title, result.url) for result in opinion_contents]}\n")
