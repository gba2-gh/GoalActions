import json
import os


class ScoreChancePlay:
    def __init__(self):

        self.include_outcome_score = True
        self.include_quality_score = True
        self.include_location_score = True



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
        "narrowly wide": 6,
        "narrowly over": 6,
        "painfully wide": 6,
        
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
        "comfortable save": 3,
        "easy save": 3,
        "comfortably saved": 3,
        "saved by the keeper": 3,
        "effort is saved": 3,
        
        # Tier 4 (Low/No Danger)
        "goes wide": 2,
        "sails well over": 2,
        "flies high over": 2,
        "well wide"
        "way wide": 2,
        "misses": 2,
        "clears the ball": 1,
        "snuffed out": 1,
        "cuts it out": 1,
        "intercepts": 1,
        "overhits the cross": 1,
        }

        self.QUALITY_SCORES = {
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

        self.LOCATION_SCORES = {
        # Location Modifiers
        "inside the box": 3,
        "penalty spot": 3,
        "penalty area": 3,
        "close range": 3,
        "goal area": 3,
        "edge of the box": 1,
        "edge of the area": 1,
        "just outside the box": 1,
        "from distance": 0,
        "long-range": 0,
        }

    def calculate_outcome_score(self, phrase):
        base_score = 0
        for keyword, score in self.OUTCOME_SCORES.items():
            if keyword in phrase:
                # We take the highest outcome score found as the base
                if score > base_score:
                    base_score = score


        return base_score

    def calculate_location_score(self, phrase):
        base_score = 0
        for keyword, score in self.LOCATION_SCORES.items():
            if keyword in phrase:
                # We take the highest outcome score found as the base
                if score > base_score:
                    base_score = score
        return base_score
    
    def calculate_quality_score(self, phrase):
        base_score = 0
        for keyword, score in self.QUALITY_SCORES.items():
            if keyword in phrase:
                # We take the highest outcome score found as the base
                base_score += score
        return base_score
    

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
       
        score = self.calculate_outcome_score(lower_phrase)
        if self.include_location_score:
            score += self.calculate_location_score(lower_phrase)
        if self.include_quality_score:
            score += self.calculate_quality_score(lower_phrase)
                
        return score

        

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
        