# Determining the similarity of a webpage's sentences using Jaccard Similarity,
# then analyzing the sentiment of the tokens, extracting opinions, concisely expressing
# them and their opposites, and querying Metaphor for up-to-date resources on it

from metaphor_python import Metaphor
from nltk import sent_tokenize, word_tokenize
import sys
import json
import openai
import os
import requests
import re
from bs4 import BeautifulSoup

# jac_threshold is used to decide whether to combine sentences as one context,
# if the jaccard similarity value >= threshold they will be combined
jac_threshold = .1

# API Details for Metaphor content and search
contents_api_base_url = "https://api.metaphor.systems/contents?ids="
headers = {
    "accept": "application/json",
    "x-api-key": os.getenv("METAPHOR_API_KEY")
}
metaphor = Metaphor(os.getenv("OPENAI_API_KEY"))

# API Details for OpenAI API
openai.api_key = os.getenv("METAPHOR_API_KEY")

# System Message for OpenAI GPT3.5 API Call
GPT_SYSTEM_MESSAGE = "You are an assistant that can help review the sentiment of a given text. " \
					"Determine whether or not it contains an opinion and express the opinion concisely. " \
					"Then, express the opposite of the opinion concisely. Include the subject of the text. " \
					"Make the output JSON formatted, with properties for the subject, sentiment, opinion, and opposite_opinion."

# Clean HTML tags from the webpage content so we can tokenize
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Jaccard Similarity, divides number of observations in both sets
# by the number of observations in either set
def jaccard(list1, list2):
	intersection = len(list(set(list1).intersection(list2)))
	union = (len(list1) + len(list2)) - intersection
	return float(intersection) / union

# Call the Metaphor content rest API endpoint with the hash that was passed in
def get_contents(hash):
    contents_api_full_url = contents_api_base_url + hash
    contents_response = requests.get(contents_api_full_url, headers=headers).text
    content_json = json.loads(contents_response)
    # Error handling in case the content is not found
    if 'error' in content_json:
        print("Invalid Metaphor content hash!")
        quit()
    html_text = content_json['contents'][0]['extract']
    webpage_text = remove_html_tags(html_text)
    return webpage_text

# Compare each sentence using Jaccard Similarity, keep track of those whose similarity crosses our threshold
# Saving sentence positions to highlight contexts in future iterations
def discover_similar_sentences(sentence_words):
    word_index = 0
    sentence_positions = {}
    similar_sentences = {}
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
    	return similar_sentences, sentence_positions

# Combine similar sentences, not including repeated similarities, filtered out with a set
def combine_contexts(similar_sentences, webpage_sentences):
    used_sentences = set()
    curr_sentence = ""
    contexts = []
    i = 0
    while i < len(webpage_sentences):
        curr_sentence = webpage_sentences[i]
        j = 0
        curr_similar_sentences = similar_sentences.get(i)

        if not curr_similar_sentences:
            i += 1
            contexts.append(curr_sentence)
            used_sentences.add(i)
            continue

        while j < len(curr_similar_sentences):
            if j in used_sentences:
    	        j += 1
    	        continue

            curr_sentence += " " + webpage_sentences[j]
            used_sentences.add(j)
            j += 1

        contexts.append(curr_sentence)
        used_sentences.add(i)
        i += 1
    return contexts

# Call OpenAI GPT3.5 API to get the opinion and its opposite
def get_opinions_openai(user_input):
    opinion_completion = openai.ChatCompletion.create(
    	model="gpt-3.5-turbo",
    	messages=[
    		{"role": "system", "content": GPT_SYSTEM_MESSAGE},
    		{"role": "user", "content": user_input},
    	],
    )

    result = json.loads(opinion_completion.choices[0].message.content)
    return result

# Search Metaphor for results about the opinion of the user selected context and its opposite
def search_opinions_metaphor(opinion, opposite_opinion):
    search_response_opinion = metaphor.search(
    	opinion, use_autoprompt=True, start_published_date="2023-06-01"
    )

    search_response_opposite = metaphor.search(
    	opposite_opinion, use_autoprompt=True, start_published_date="2023-06-01"
    )
    opinion_contents = search_response_opinion.results
    opposite_contents = search_response_opposite.results
    return opinion_contents, opposite_contents

# Error handling, will quit if there is not a content hash
if len(sys.argv) < 2:
	print("Please provide an argument containing a document's Metaphor hash!")
	quit()

# Pass in the hash argument and get the content from the Metaphor content API endpoint
webpage_hash = sys.argv[1]
webpage_text = get_contents(webpage_hash)
webpage_sentences = sent_tokenize(webpage_text)
sentence_words = []

# Add tokenized words to array of each sentence
for i in range(len(webpage_sentences)):
	sentence_words.append(word_tokenize(webpage_sentences[i]))

# Match up similar sentences using Jaccard Similarity threshold
similar_sentences, sentence_positions = discover_similar_sentences(sentence_words)
# Combine the sentences that were matched using Jaccard Similarity
contexts = combine_contexts(similar_sentences, webpage_sentences)

# Allow user to pick content to explore
i = 0
while i < len(contexts):
	print(f"{i}. " + contexts[i])
	i += 1

print(f"Here are each of the article's sentences organized by their context. \n"
						f"Please select one of them to learn more about it:\n")
selection = input()
# Error handling to ensure the selection is valid
while not selection.isdigit():
	print("Please enter the number for your selection, using an integer.")
	selection = input()

user_context_input = contexts[int(selection)]

# Call OpenAI GPT3.5 API to get opinion and its opposite in JSON format
openai_opinion_result = get_opinions_openai(user_context_input)

# Get each of the OpenAI opinion values from the dictionary
subject = openai_opinion_result["subject"]
sentiment = openai_opinion_result["sentiment"]
opinion = openai_opinion_result["opinion"]
opposite_opinion = openai_opinion_result["opposite_opinion"]

# Search Metaphor for results about the opinion and its opposite
opinion_contents, opposite_contents = search_opinions_metaphor(opinion, opposite_opinion)

# Display the results from Metaphor
print(f"More opinions like this can be found at these webpages: "
						f"{[(result.title, result.url) for result in opinion_contents]}\n")

print(f"Find out more about the opposite of this opinion here: "
						f"{[(result.title, result.url) for result in opposite_contents]}\n")
