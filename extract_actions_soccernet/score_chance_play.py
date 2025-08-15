import json
import os


class ScoreChancePlay:
    def __init__(self):

        # Keys should be lowercase for case-insensitive matching.
        # Relevant Action keywords extracted with Gemini
        self.OUTCOME_SCORES = {
        # Tier 1 (Highest Danger)
        "crossbar": 10,
        "post": 10,
        "woodwork": 10,
        "brilliant save": 8,
        "astonishing save": 8,
        "amazing save": 8,
        "remarkable save": 8,
        "superb save": 8,
        "stunning save": 8,
        "inches wide": 7,
        "inches over": 7,
        "whisker": 7,
        "narrowly wide": 7,
        "narrowly over": 7,
        "painfully wide": 7,
        
        # Tier 2 (Dangerous)
        "decent save": 5,
        "fine save": 5,
        "good save": 5,
        "denies him": 5, 
        "keeps it out": 5,
        "is equal to it": 5,
        "blocked": 4,
        "block": 4,
        
        # Tier 3 (Moderate Danger)
        "comfortable save": 2,
        "easy save": 2,
        "comfortably saved": 2,
        "saved by the keeper": 2,
        "effort is saved": 2,
        
        # Tier 4 (Low/No Danger)
        "goes wide": 0,
        "sails well over": 0,
        "flies high over": 0,
        "way wide": 0,
        "misses": 0,
        "clears the ball": 0,
        "snuffed out": 0,
        "cuts it out": 0,
        "intercepts": 0,
        "overhits the cross": 0,
        }

        self.MODIFIER_SCORES = {
        # Location Modifiers
        "inside the box": 3,
        "penalty spot": 3,
        "penalty area": 3,
        "close range": 3,
        "goal area": 3,
        "edge of the box": 1,
        "edge of the area": 1,
        "just outside the box": 1,
        "18 metres": 1,
        "from distance": 0,
        "long-range": 0,
        
        # Quality Modifiers (Positive)
        "great opportunity": 2,
        "golden chance": 2,
        "perfect cross": 2,
        "great touch": 2,
        "unmarked": 2,
        "sweet pass": 2,
        "pin-point cross": 2,
        
        # Quality Modifiers (Negative)
        "poorly": -2,
        "poor effort": -2,
        "bad attempt": -2,
        "lacks power": -2,
        "doesn't hit properly": -2,
        "not at all pretty": -2,
        "wastes": -2,
        }


    def calculate_danger_score(self, phrase: str) -> int:
        """
        Calculates a 'danger score' for a football action phrase based on predefined keywords.

        Args:
            phrase: A string describing a football action.

        Returns:
            An integer score representing the level of danger.
        """
        # Convert phrase to lowercase for case-insensitive matching
        lower_phrase = phrase.lower()
        
        base_score = 0
        for keyword, score in self.OUTCOME_SCORES.items():
            if keyword in lower_phrase:
                # We take the highest outcome score found as the base
                if score > base_score:
                    base_score = score
        
        modifier_score = 0
        for keyword, score in self.MODIFIER_SCORES.items():
            if keyword in lower_phrase:
                modifier_score += score # in this case, only 1 modifier score
                
        return base_score + modifier_score

        

if __name__ == "__main__":

    with open('results/all_plays_MT.json', mode='r', encoding='utf-8') as f:
        plays = json.load(f)

    scorer = ScoreChancePlay()
    scored_phrases = []
    for play in plays:
        #Score only no goals
        if play['play_is_goal'] == False:
            score = scorer.calculate_danger_score(play['caption'])
            scored_phrases.append((score, play['caption']))

    # Sort the phrases from most to least dangerous
    scored_phrases.sort(key=lambda x: x[0], reverse=True)

    # Output results
    output_path = 'results/score_close_chances'
    os.makedirs(output_path, exist_ok=True)
    with open(f'{output_path}/danger_scores.json', mode='w', encoding='utf-8') as f:
            json.dump(scored_phrases, f, indent=2)
        