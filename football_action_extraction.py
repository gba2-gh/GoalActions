import ollama
import json
import re
import pandas as pd


class FootballActionExtractor:
    def __init__(self):
        self.keywords = ['pass', 'cross', 'shot', 'take_on', 'dribble', 'interception', 'tackle', 'keeper_save', 'free_kick', 'corner']

        self.ACTION_KEYWORDS = {
            'pass': [
                'pass', 'passes', 'passed', 'feeds', 'finds', 'provides', 'square',
                'precise pass', 'sweet pass', 'latch pass', 'defence-splitting pass', 'low pass'
            ],
            'cross': [
                'cross', 'crosses', 'crossed', 'delivery', 'whips', 'swings in', 
                'ball into the area', 'ball into the box', 'perfect cross', 
                'inch-perfect cross', 'score cross'
            ],
            'shot': [
                'shot', 'shoots', 'effort', 'strike', 'finish', 'finishes', 'scores', 
                'goal', 'attempts', 'unleashes', 'pulls trigger', 'flies wide', 
                'towards goal', 'flashes inches', 'promising distance', 'header'
            ],
            'take_on': [
                'beats', 'past', 'dribbles past', 'takes on', 'goes past', 
                'nutmeg', 'skill'
            ],
            'dribble': [
                'dribble', 'dribbles', 'run', 'carries', 'advances', 
                'drives forward', 'maneuver', 'control', 'solo run'
            ],
            'interception': [
                'intercept', 'intercepts', 'blocks', 'blocked', 'cuts out', 
                'steals', 'wins', 'deflected'
            ],
            'tackle': [
                'tackle', 'tackles', 'challenges', 'dispossess', 'wins the ball'
            ],
            'keeper_save': [
                'save', 'saves', 'keeper', 'goalkeeper', 'stops', 'parries', 
                'catches', 'stunning save', 'superb save', 'stop effort', 'keeps ball'
            ],
            'free_kick': [
                'free kick', 'frees', 'set piece', 'direct free kick', 
                'placed ball', 'spot kick'
            ],
            'corner': [
                'corner', 'corner kick', 'from the corner', 'flag kick', 'corner ball', 
                'swings in corner'
            ]
        }

        # System prompt for Ollama
        SYSTEM_PROMPT = f"""
        You are a football (soccer) action analyzer. 

        Your task is to extract the sequence of football actions from commentary text.
        Follow these rules:
        1. Extract actions in TRUE chronological order (ignore retrospective commentary).
        2. Use ONLY these actions: {", ".join(self.ACTION_KEYWORDS.keys())}.
        3. Do not duplicate repeated mentions of the same action.
        4. Return ONLY a JSON array of actions, nothing else.
        """

    def keyword_extract(self, text):
        """
        Rule-based extraction of actions using keyword matching.
        Preserves chronological order by scanning text left-to-right.
        """
        found_actions = []
        lowered = text.lower()
        for action, keywords in self.ACTION_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", lowered):
                    if action not in found_actions:
                        found_actions.append(action)
                    break  # stop after first keyword match for this action
        return found_actions

    def ollama_extract(self, text, model="llama3.1"):
        """
        Use Ollama to extract actions.
        """
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f'Text to analyze: "{text}"'}
            ]
        )
        print(response)
        try:
            actions = json.loads(response["message"]["content"])
        except Exception:
            actions = []
        return actions

    def analyze_commentaries(self, text, model="llama3.1"):
        """
        Run both Ollama and keyword search on multiple commentary texts.
        """
        #results = []
        #for text in entries:
        #ollama_actions = ollama_extract(text, model)
        ollama_actions = None
        keyword_actions = self.keyword_extract(text)
        results = {
            "text": text,
            "ollama_actions": ollama_actions,
            "keyword_actions": keyword_actions
        }
        return results


    def extract_from_dataframe(self, dataframe):

        for index,play in dataframe.iterrows():
            
            print(f"Processing {play['id']}")

            c_result = self.analyze_commentaries(play['caption'])
            n_result = self.analyze_commentaries(play['narration'])
            result = {'id' : play['id'],
                    'caption_llama': c_result['ollama_actions'],
                    'caption_rule': c_result['keyword_actions'],
                    'narrated_llama': n_result['ollama_actions'],
                    'narrated_rule': n_result['keyword_actions'],
            }

            vector = [1 if kw in c_result['keyword_actions'] else 0 for kw in self.keywords]

            play['actions_vector'] = vector 

        return dataframe
    
    def extract_from_dict(self, all_plays):

        for play in all_plays:
            
            print(f"Processing {play['id']}")

            c_result = self.analyze_commentaries(play['caption'])
            n_result = self.analyze_commentaries(play['narration'])
            result = {'id' : play['id'],
                    'caption_llama': c_result['ollama_actions'],
                    'caption_rule': c_result['keyword_actions'],
                    'narrated_llama': n_result['ollama_actions'],
                    'narrated_rule': n_result['keyword_actions'],
            }

            vector = [1 if kw in c_result['keyword_actions'] else 0 for kw in self.keywords]

            play['actions_vector'] = vector 

        return all_plays



def main():
    

    with open('results/all_plays.json', mode='r', encoding='utf-8') as f:
        all_plays = json.load(f)

    results = []

    extractor = FootballActionExtractor()

    for play in all_plays:
        
        print(f"Processing {play['id']}")

        c_result = extractor.analyze_commentaries(play['caption'])
        n_result = extractor.analyze_commentaries(play['narration'])
        result = {'id' : play['id'],
                  'caption_llama': c_result['ollama_actions'],
                  'caption_rule': c_result['keyword_actions'],
                  'narrated_llama': n_result['ollama_actions'],
                  'narrated_rule': n_result['keyword_actions'],
        }

        results.append(result)

    
        vector = [1 if kw in c_result['keyword_actions'] else 0 for kw in extractor.keywords]

        play['actions_vector'] = vector 
        
        #print(results)
        

    with open('results/football_action_extraction/simple.json', mode='w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    #print(json.dumps(results, indent=2))
    df = pd.DataFrame(results)
    df.to_csv('results/football_action_extraction/simple.csv', index=False)

    with open('results/all_plays_vector.json', mode='w', encoding='utf-8') as f:
        json.dump(all_plays, f, indent=2)


if __name__ == "__main__":
    main()