import ollama
import re
import json
from typing import List, Dict, Any
import os
import pandas

class FootballActionExtractor:
    def __init__(self, model_name: str = "llama3.1"):
        """
        Initialize the extractor with an Ollama model
        
        Args:
            model_name: Name of the Ollama model to use (e.g., 'llama3.1', 'mistral', 'codellama')
        """
        self.model_name = model_name
        self.client = ollama.Client()
        
        # Define action mappings and keywords
        self.action_keywords = {
            'pass': ['pass', 'passes', 'passed', 'cross', 'crosses', 'crossed', 'delivery', 'feeds', 'finds', 'provides', 'square'],
            'cross': ['cross', 'crosses', 'crossed', 'delivery', 'whips', 'swings in', 'ball into the area', 'ball into the box'],
            'shot': ['shot', 'shoots', 'effort', 'strike', 'finish', 'finishes', 'scores', 'goal', 'attempts'],
            'take_on': ['beats', 'past', 'dribbles past', 'takes on', 'goes past', 'nutmeg', 'skill'],
            'dribble': ['dribble', 'dribbles', 'run', 'carries', 'advances', 'drives forward', 'maneuver', 'control'],
            'interception': ['intercept', 'intercepts', 'blocks', 'blocked', 'cuts out', 'steals', 'wins'],
            'tackle': ['tackle', 'tackles', 'challenges', 'dispossess', 'wins the ball'],
            'keeper_save': ['save', 'saves', 'keeper', 'goalkeeper', 'stops', 'parries', 'catches'],
            'free_kick' : ['free kick'],
            'corner': ['corner']
        }
    
    def preprocess_narrated_text(self, narrated_text: str) -> str:
        """
        Clean and preprocess narrated caption text
        
        Args:
            narrated_text: Raw narrated text with timestamps
            
        Returns:
            Cleaned text without timestamps
        """
        # Remove timestamp brackets [xxx, yyy]
        cleaned = re.sub(r'\[[\d.,\s]+\]', '', narrated_text)
        
        # Join sentences and clean up
        sentences = cleaned.split(';')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.endswith('.'):
                sentence += '.'
            if sentence:
                cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)
    
    def create_extraction_prompt(self, text: str, is_narrated: bool = False) -> str:
        """
        Create a prompt for the LLM to extract actions
        
        Args:
            text: Football commentary text
            is_narrated: Whether this is narrated commentary (may contain retrospective analysis)
            
        Returns:
            Formatted prompt for action extraction
        """
        context_instruction = ""
        if is_narrated:
            context_instruction = """
IMPORTANT CONTEXT: This is live football commentary that may include retrospective analysis after a goal. 
The commentary often:
- Describes actions as they happen in real-time
- Then replays/analyzes the same sequence after a goal
- May jump back and forth in time explaining the build-up

Your task is to identify the ACTUAL CHRONOLOGICAL SEQUENCE of actions that led to the goal or key moment, 
not just process the text linearly. Look for:
- The actual sequence of events that built up to the goal
- Ignore repeated descriptions of the same actions
- Focus on the logical flow: how did the ball move from player to player?
- If the same action is described multiple times, count it only once in the correct chronological position
"""
        
        prompt = f"""
            You are a football (soccer) action analyzer. Extract the sequence of actions from the following football commentary text.
            {context_instruction}

            Available actions and their definitions:
            - pass: normal pass in open play
            - cross: cross into the box  
            - shot: shot attempt (including goals)
            - take_on: dribble past opponent
            - dribble: player dribbles at least 3 meters with the ball
            - interception: interception of pass or defensive block of shot
            - tackle: tackle on the ball
            - keeper_save: keeper saves a shot on goal
            - free_kick: player plays the ball from free kick
            - corner: player playes the ball from corner

            Text to analyze: "{text}"

            Instructions:
            1. Understand the ACTUAL chronological sequence of events, not just the text order
            2. If this is commentary with retrospective analysis, reconstruct the real sequence
            3. Identify each unique football action that occurred in the play
            4. Return ONLY a JSON array of actions in their TRUE chronological order
            5. Use only the action names listed above
            6. Avoid duplicates - if the same action is described multiple times, include it only once
            7. Focus on the logical flow of the ball and key player actions

            Example scenarios:
            - "Hazard passes to Costa, great goal! Let me replay that - Hazard with the pass, then Costa scores" 
            → ["pass", "shot"] (not ["pass", "shot", "pass", "shot"])
            - "What a save! The keeper stops it. Earlier, Messi had dribbled past two defenders before shooting"
            → ["dribble", "take_on", "shot", "keeper_save"]

            Output:"""
        
        return prompt
    
    def extract_actions_with_llm(self, text: str, is_narrated:bool = False) -> List[str]:
        """
        Use Ollama LLM to extract actions from text
        
        Args:
            text: Football commentary text
            
        Returns:
            List of extracted actions
        """
        try:
            prompt = self.create_extraction_prompt(text)
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.1,  # Low temperature for consistent results
                    'top_p': 0.9,
                }
            )
            
            # Extract the response content
            response_text = response['message']['content'].strip()

            return response_text
            
            # Try to parse JSON from response
            try:
                # Look for JSON array in the response
                json_match = re.search(r'\[.*?\]', response_text)
                if json_match:
                    actions = json.loads(json_match.group())
                    return [action.lower().replace(' ', '_') for action in actions if isinstance(action, str)]
                else:
                    # Fallback: try to parse the entire response as JSON
                    actions = json.loads(response_text)
                    return [action.lower().replace(' ', '_') for action in actions if isinstance(action, str)]
            
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from response: {response_text}")
                return self.fallback_extraction(text)
                
        except Exception as e:
            #print(f"Error with Ollama: {e}")
            return self.fallback_extraction(text)
    
    def fallback_extraction(self, text: str) -> List[str]:
        """
        Fallback keyword-based extraction if LLM fails
        
        Args:
            text: Football commentary text
            
        Returns:
            List of extracted actions using keyword matching
        """
        text_lower = text.lower()
        actions = []
        
        # Simple keyword matching as fallback
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue
                
            # Check for each action type
            for action_type, keywords in self.action_keywords.items():
                if any(keyword in sentence for keyword in keywords):
                    actions.append(action_type)
                    break
        
        return actions
    
    def extract_from_goal_caption(self, caption: str) -> List[str]:
        """
        Extract actions from goal caption format
        
        Args:
            caption: Goal caption text
            
        Returns:
            List of extracted actions
        """
        return self.extract_actions_with_llm(caption, is_narrated=False)
    
    def extract_from_narrated_caption(self, narrated_caption: str) -> List[str]:
        """
        Extract actions from narrated caption format
        
        Args:
            narrated_caption: Narrated caption with timestamps
            
        Returns:
            List of extracted actions
        """
        cleaned_text = self.preprocess_narrated_text(narrated_caption)
        return self.extract_actions_with_llm(cleaned_text, is_narrated=True)
    
    def analyze_narrative_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the structure of narrated commentary to understand context
        
        Args:
            text: Commentary text
            
        Returns:
            Dictionary with analysis of narrative structure
        """
        analysis_prompt = f"""
            Analyze this football commentary to understand its narrative structure:

            Text: "{text}"

            Identify:
            1. Is there a goal or key moment described?
            2. Are there retrospective descriptions (replaying events after they happened)?
            3. What is the likely chronological sequence of events?
            4. Are there any temporal indicators (e.g., "earlier", "before that", "let me replay")?

            Provide a brief analysis focusing on the narrative flow and whether events are described chronologically or with flashbacks.

            Analysis:"""
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': analysis_prompt}],
                options={'temperature': 0.3}
            )
            
            return {
                'has_retrospective': 'replay' in response['message']['content'].lower() or 'earlier' in response['message']['content'].lower(),
                'analysis': response['message']['content']
            }
        except:
            return {'has_retrospective': True, 'analysis': 'Could not analyze structure'}
    
    def extract_with_context_awareness(self, text: str, is_narrated: bool = False) -> Dict[str, Any]:
        """
        Extract actions with full context awareness and analysis
        
        Args:
            text: Commentary text
            is_narrated: Whether this is narrated commentary
            
        Returns:
            Dictionary with LLM actions, rule-based actions, context analysis, and comparison
        """
        result = {
            'llm_actions': [],
            'rule_based_actions': [],
            'narrative_analysis': None,
            'confidence': 'high',
            'methods_agree': False,
            'agreement_ratio': 0.0
        }
        
        # Always get rule-based extraction as baseline
        result['rule_based_actions'] = self.fallback_extraction(text)
        
        if is_narrated:
            # First analyze the narrative structure
            result['narrative_analysis'] = self.analyze_narrative_structure(text)
            
            # Use enhanced extraction with context awareness
            enhanced_prompt = f"""
            You are analyzing football commentary that may contain retrospective analysis. 

            NARRATIVE CONTEXT: {result['narrative_analysis']['analysis']}

            Now extract the ACTUAL chronological sequence of actions from this text:
            "{text}"

            Focus on reconstructing the true sequence of events, ignoring repeated descriptions and retrospective analysis.

            Available actions: pass, cross, shot, take_on, dribble, interception, tackle, keeper_save

            Return only a JSON array of the chronological action sequence: """
            
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': enhanced_prompt}],
                    options={'temperature': 0.1}
                )
                
                response_text = response['message']['content'].strip()

                #test
                result['llm_actions'] = response_text
                return result

                json_match = re.search(r'\[.*?\]', response_text)
                if json_match:
                    actions = json.loads(json_match.group())
                    result['llm_actions'] = [action.lower().replace(' ', '_') for action in actions if isinstance(action, str)]
                else:
                    result['llm_actions'] = self.extract_actions_with_llm(text, is_narrated=True)
                    result['confidence'] = 'medium'
                    
            except Exception as e:
                result['llm_actions'] = self.extract_actions_with_llm(text, is_narrated=True)
                result['confidence'] = 'low'
        else:
            result['llm_actions'] = self.extract_actions_with_llm(text, is_narrated=False)
        
        # Compare LLM and rule-based results
        result['methods_agree'] = result['llm_actions'] == result['rule_based_actions']
        
        # Calculate agreement ratio (percentage of matching actions)
        if result['llm_actions'] and result['rule_based_actions']:
            # Simple overlap calculation
            llm_set = set(result['llm_actions'])
            rule_set = set(result['rule_based_actions'])
            overlap = len(llm_set.intersection(rule_set))
            total_unique = len(llm_set.union(rule_set))
            result['agreement_ratio'] = overlap / total_unique if total_unique > 0 else 0.0
        
        return result
    
    def extract_all_methods(self, text: str, is_narrated: bool = False) -> Dict[str, Any]:
        """
        Extract actions using all available methods for comprehensive comparison
        
        Args:
            text: Commentary text
            is_narrated: Whether this is narrated commentary
            
        Returns:
            Dictionary with results from all extraction methods
        """
        result = {
            'text': text,
            'is_narrated': is_narrated,
            'methods': {
                'context_aware_llm': [],
                'standard_llm': [],
                'rule_based': []
            },
            'comparison': {
                'all_methods_agree': False,
                'llm_vs_rules_agree': False,
                'context_vs_standard_agree': False
            }
            
        }
        
        # Get rule-based extraction
        result['methods']['rule_based'] = self.fallback_extraction(text)
        
        # Get standard LLM extraction
        result['methods']['standard_llm'] = self.extract_actions_with_llm(text, is_narrated=False)
        
        # Get context-aware extraction
        if is_narrated:
            context_result = self.extract_with_context_awareness(text, is_narrated=True)
            result['methods']['context_aware_llm'] = context_result['llm_actions']
            result['narrative_analysis'] = context_result['narrative_analysis']
        else:
            result['methods']['context_aware_llm'] = self.extract_actions_with_llm(text, is_narrated=False)
        
        # Compare all methods
        methods = result['methods']
        result['comparison']['all_methods_agree'] = (
            methods['context_aware_llm'] == methods['standard_llm'] == methods['rule_based']
        )
        result['comparison']['llm_vs_rules_agree'] = (
            methods['context_aware_llm'] == methods['rule_based']
        )
        result['comparison']['context_vs_standard_agree'] = (
            methods['context_aware_llm'] == methods['standard_llm']
        )
        

        
        return result
    
    def batch_extract(self, captions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract actions from multiple captions
        
        Args:
            captions: List of caption dictionaries with 'text' and 'type' keys
            
        Returns:
            List of results with original text and extracted actions
        """
        results = []
        
        for caption_data in captions:
            text = caption_data.get('text', '')
            caption_type = caption_data.get('type', 'goal')  # 'goal' or 'narrated'
            
            if caption_type == 'narrated':
                actions = self.extract_from_narrated_caption(text)
            else:
                actions = self.extract_from_goal_caption(text)
            
            results.append({
                'original_text': text,
                'type': caption_type,
                'extracted_actions': actions
            })
        
        return results


# Example usage
def main():
    # Initialize the extractor
    extractor = FootballActionExtractor(model_name="llama3.1")  # Change model as needed

    
    # Test individual extractions with context awareness
    print("=== Goal Caption Extraction ===")
    goal_result = extractor.extract_with_context_awareness(goal_caption, is_narrated=False)
    print(f"Extracted actions: {goal_result['llm_actions']}")
    print(f"Confidence: {goal_result['confidence']}")
    
    print("\n=== Narrated Caption Extraction (Context-Aware) ===")
    narrated_result = extractor.extract_with_context_awareness(narrated_caption, is_narrated=True)
    print(f"Extracted actions: {narrated_result['llm_actions']}")
    print(f"Confidence: {narrated_result['confidence']}")
    if narrated_result['narrative_analysis']:
        print(f"Narrative analysis: {narrated_result['narrative_analysis']['analysis'][:200]}...")
    
    # Test traditional extraction for comparison
    print("\n=== Traditional Extraction (for comparison) ===")
    goal_actions = extractor.extract_from_goal_caption(goal_caption)
    narrated_actions = extractor.extract_from_narrated_caption(narrated_caption)
    print(f"Goal actions (traditional): {goal_actions}")
    print(f"Narrated actions (traditional): {narrated_actions}")
    
    # Test batch processing
    print("\n=== Batch Processing ===")
    batch_data = [
        {'text': goal_caption, 'type': 'goal'},
        {'text': narrated_caption, 'type': 'narrated'}
    ]
    
    results = extractor.batch_extract(batch_data)
    for i, result in enumerate(results):
        print(f"Caption {i+1} ({result['type']}): {result['extracted_actions']}")




if __name__ == "__main__":
    #main()
    goal_caption = """Goal! Eden Hazard provides Branislav Ivanovic (Chelsea) with a nice pass inside the box. 
    It allows him to finish with a precise effort into the bottom right corner. 1:0. 
    """
    
    narrated_caption = """ [781.04, 785.04], The South American who definitely steals that ball is still fighting there.;
      [785.04, 786.04], Play again for Hazard.; [787.04, 788.04], Already inside the area.;
        [788.04, 790.04], What a fantastic maneuver by the Belgian.;
          [790.04, 791.04], Ivanovic.; [791.04, 792.04], Ivanovic scores again.;
           [793.04, 794.04], Well, Ivanovic scores again.; 
           [794.04, 797.04], Although the player is from Eden Hazard.; 
           [798.04, 800.04], Minute 14 of the first half.; 
           [801.04, 803.04], Perhaps Barley is too soft in defense.; 
           [803.04, 805.04], But great individual action from Hazard.;
             [806.04, 809.04], Ivanovic once again faithful to his appointment with the goal.;
               [809.04, 811.04], Score the first for Chelsea 1-0.; 
               [812.04, 814.04], Yes, the Barley is going to be soft here.;
                 [814.04, 817.04], With that excessively easy turnover.; 
                 [817.04, 820.04], Then Hazard's thing is impressive.;
"""


    extractor = FootballActionExtractor(model_name="llama3.1")  # Change model as needed
    with open('results/all_plays.json', mode='r', encoding='utf-8') as f:
        all_plays = json.load(f)

    results = []
    for play in all_plays:
        # if play['id'] >10:
        #     break
        print(f"\n Processing caption {play['id']} \n")
        caption_result = extractor.extract_all_methods( play['caption'], is_narrated = False) 
        narrated_result = extractor.extract_all_methods( play['narration'], is_narrated = True) 

        caption_method = caption_result['methods']
        #comparison_result = caption_result['comparison']
        narrated_method = narrated_result['methods']

        result = {'id': play['id'],
                  'caption': play['caption'],

                  'caption_context_aware_llm': caption_method['context_aware_llm'],
                  'caption_standard_llm': caption_method['standard_llm'],
                  'caption_rule_based': caption_method['rule_based'],
                  'narration': play['narration'],
                  'narrated_narrative_analysis': narrated_result['narrative_analysis'],
                    'narrated_context_aware_llm': narrated_method['context_aware_llm'],
                  'narrated_standard_llm': narrated_method['standard_llm'],
                  'narrated_rule_based': narrated_method['rule_based'],
                    }
        results.append(result)
        
    print('Saving results')
    os.makedirs('results/football_action_extraction', exist_ok=True)
    with open('results/football_action_extraction/action_extraction.json', mode='w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    df = pandas.DataFrame(results)
    df.to_csv('results/football_action_extraction/action_extraction.csv', index=False)


    #print(caption_result)
        
        # print(method_result['context_aware_llm'])
        # print(method_result['standard_llm'])
        # print(method_result['rule_based'])
    



            # 'text': text,
            # 'is_narrated': is_narrated,
            # 'methods': {
            #     'context_aware_llm': [],
            #     'standard_llm': [],
            #     'rule_based': []
            # },
            # 'comparison': {
            #     'all_methods_agree': False,
            #     'llm_vs_rules_agree': False,
            #     'context_vs_standard_agree': False