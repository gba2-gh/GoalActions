import re
from collections import Counter
from itertools import islice
import nltk
from nltk.corpus import stopwords
import json

with open('results/all_plays.json', mode='r', encoding='utf-8') as f:
        all_plays = json.load(f)



# Input text
text = ""
for play in all_plays:
    text = text + '; ' + play['narration']


# Download stopwords (only once)
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Add custom football stopwords
football_fillers = {"here", "there", "its", "he", "hes", "what", "well"}
stop_words.update(football_fillers)

# Your ACTION_KEYWORDS dictionary from earlier
ACTION_KEYWORDS = {
    'pass': ['pass', 'passes', 'passed', 'feeds', 'finds', 'provides', 'square'],
    'cross': ['cross', 'crosses', 'crossed', 'delivery', 'whips', 'swings in', 'ball into the area', 'ball into the box'],
    'shot': ['shot', 'shoots', 'effort', 'strike', 'finish', 'finishes', 'scores', 'goal', 'attempts', 'header'],
    'take_on': ['beats', 'past', 'dribbles past', 'takes on', 'goes past', 'nutmeg', 'skill'],
    'dribble': ['dribble', 'dribbles', 'run', 'carries', 'advances', 'drives forward', 'maneuver', 'control'],
    'interception': ['intercept', 'intercepts', 'blocks', 'blocked', 'cuts out', 'steals', 'wins'],
    'tackle': ['tackle', 'tackles', 'challenges', 'dispossess', 'wins the ball'],
    'keeper_save': ['save', 'saves', 'keeper', 'goalkeeper', 'stops', 'parries', 'catches']
}

# Flatten action keywords into one list
action_terms = [kw for kws in ACTION_KEYWORDS.values() for kw in kws]


# Step 1: Remove timestamps & lowercase
clean_text = re.sub(r"\[\d+.\d+,\s*\d+.\d+\],", "", text)
clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", clean_text).lower()

# Step 2: Tokenize & remove stopwords
tokens = [t for t in clean_text.split() if t not in stop_words]

# Step 3: Count action-related words only
action_counts = Counter([t for t in tokens if t in action_terms])

# Step 4: Extract context windows around action terms
context_phrases = []
window_size = 2
for i, token in enumerate(tokens):
    if token in action_terms:
        start = max(0, i - window_size)
        end = i + window_size + 1
        phrase = " ".join(tokens[start:end])
        context_phrases.append(phrase)

context_counts = Counter(context_phrases)

# Results
print("\nTop Action Words:")
for word, count in action_counts.most_common(25):
    print(f"{word}: {count}")

print("\nTop Context Phrases:")
for phrase, count in context_counts.most_common(25):
    print(f"{phrase}: {count}")

