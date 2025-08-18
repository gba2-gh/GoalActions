# SoccerNet Analysis with DeepSeek API - Batch Processing Implementation
import os
import json
import re
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class DeepSeekConfig:
    api_key: str
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com/v1/chat/completions"
    batch_size: int = 5
    num_workers: int = 2
    max_tokens: int = 4000  # Increased for batch responses
    temperature: float = 0.1

class BatchDeepSeekAnalyzer:
    """
    DeepSeek analyzer with multiple batch processing strategies
    """
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.request_count = 0
        self.cache = {}  # Simple cache for repeated descriptions
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"DeepSeek analyzer initialized:")
        print(f"   Model: {config.model}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Workers: {config.num_workers}")
    
    # ========================================
    # STRATEGY 1: Multi-Event Single Prompt
    # ========================================
    def analyze_batch_single_prompt(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Analyze multiple events in a single prompt for efficiency
        """
        if not events:
            return []
        
        # Build batch prompt
        events_text = ""
        for i, event in enumerate(events):
            events_text += f"### Event {i+1} ###\n"
            events_text += f"Time: {event['game_time']}\n"
            events_text += f"Description: {event['description']}\n\n"
        
        prompt = f"""Analyze these {len(events)} soccer events and return a JSON array with exactly {len(events)} objects.

Match: {context['home_team']} (home) vs {context['away_team']} (away)

Events:
{events_text}

Rules:
- is_goal: true only if ball crossed goal line for a score  
- is_shooting_action: true for shots, headers, volleys that could score
- is_penalty: true only if penalty kick mentioned
- team: "home"/"away"/null based on which team's player
- danger_level: 1-5 (5 = most dangerous) for non-goal shooting actions; 0 otherwise
- confidence: 0.0-1.0 how certain you are

Return ONLY a JSON array with {len(events)} objects like this:
[
  {{"is_goal": boolean, "is_shooting_action": boolean, "is_penalty": boolean, "team": "home"/"away"/null, "danger_level": int, "confidence": float}},
  ... // {len(events)} objects total
]"""

        try:
            print(f"    Batch analyzing {len(events)} events...")
            
            response = requests.post(
                self.config.base_url,
                headers=self.headers,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature
                },
                timeout=120  # Longer timeout for batches
            )
            
            if response.status_code != 200:
                print(f"API error {response.status_code}: {response.text}")
                return [self._default_response() for _ in events]
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            # Parse batch response
            parsed_results = self._parse_batch_response(response_text, len(events))
            self.request_count += 1
            
            print(f"    Batch completed: {len(parsed_results)} results")
            return parsed_results
            
        except Exception as e:
            print(f"    Batch error: {e}")
            return [self._default_response() for _ in events]
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[Dict]:
        """Parse batch response containing JSON array"""
        try:
            # Clean the response
            response_text = response_text.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])
            
            # Look for JSON array pattern
            array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            
            if array_match:
                try:
                    json_array = json.loads(array_match.group(0))
                    if isinstance(json_array, list):
                        results = []
                        for item in json_array:
                            if isinstance(item, dict):
                                results.append(self._validate_response(item))
                            else:
                                results.append(self._default_response())
                        
                        # Ensure we have the right number of results
                        while len(results) < expected_count:
                            results.append(self._default_response())
                        
                        return results[:expected_count]
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
            
            # Fallback: look for individual JSON objects
            json_objects = re.findall(r'\{[^{}]*\}', response_text)
            results = []
            for obj_text in json_objects:
                try:
                    # Extract just the JSON object part
                    obj_match = re.search(r'\{.*\}', obj_text, re.DOTALL)
                    if obj_match:
                        parsed = json.loads(obj_match.group(0))
                        results.append(self._validate_response(parsed))
                except:
                    continue
            
            # Fill to expected count
            while len(results) < expected_count:
                results.append(self._default_response())
                
            return results[:expected_count]
            
        except Exception as e:
            print(f"    Batch parse error: {e}")
            return [self._default_response() for _ in range(expected_count)]
    
    # ========================================
    # STRATEGY 2: Parallel Processing
    # ========================================
    def analyze_batch_parallel(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Process events in parallel using multiple threads
        """
        if not events:
            return []
        
        results = [None] * len(events)
        
        def process_single_event(index_event_tuple):
            index, event = index_event_tuple
            result = self.analyze_single_event(event, context)
            return index, result
        
        print(f"    Parallel processing {len(events)} events with {self.config.num_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_event, (i, event)): i 
                for i, event in enumerate(events)
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                    completed += 1
                    
                    if completed % 5 == 0:
                        print(f"      Progress: {completed}/{len(events)} events completed")
                except Exception as e:
                    print(f"Error processing event: {e}")
        
        print(f"    Parallel processing completed: {len(results)} results")
        return results
    
    # ========================================
    # STRATEGY 3: Conservative Pre-filtering + LLM Analysis
    # ========================================
    def analyze_batch_conservative_filter(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Conservative pre-filtering - only filter out obvious non-game events
        Let LLM handle all actual game situations
        """
        llm_events = []
        results = []
        
        # Conservative filtering - only remove clear administrative events
        obvious_non_game_events = [
            'match official', 'referee', 'var check', 'video assistant',
            'technical area', 'bench', 'coaching staff', 'medical staff',
            'halftime', 'full-time', 'kickoff', 'match delay',
            'weather condition', 'pitch condition', 'stadium announcement'
        ]
        
        for i, event in enumerate(events):
            desc_lower = event['description'].lower()
            
            # Only filter out clear administrative/non-game events
            if any(admin_term in desc_lower for admin_term in obvious_non_game_events):
                print(f"    Filtered obvious non-game event: '{event['description'][:50]}...'")
                results.append((i, {
                    'is_goal': False,
                    'is_shooting_action': False,
                    'is_penalty': False,
                    'team': None,
                    'danger_level': 0,
                    'confidence': 0.99  # Very high confidence this isn't a shooting action
                }))
                continue
            
            # Send everything else to LLM - let it decide
            llm_events.append((i, event))
        
        filtered_count = len(events) - len(llm_events)
        print(f"    Conservative filter: {filtered_count} obvious non-game events filtered, {len(llm_events)} events for LLM analysis")
        
        # Batch process remaining events with LLM
        if llm_events:
            indices, events_to_process = zip(*llm_events)
            llm_results = self.analyze_batch_single_prompt(list(events_to_process), context)
            
            for idx, result in zip(indices, llm_results):
                results.append((idx, result))
        
        # Sort results by original order and return
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    # ========================================
    # STRATEGY 4: LLM-First Two-Pass Analysis
    # ========================================
    def analyze_batch_two_pass(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Two-pass analysis: 
        1. Quick LLM scan to identify potential shooting events
        2. Detailed LLM analysis of identified events
        """
        if not events:
            return []
        
        print(f"    Two-pass analysis: Step 1 - Quick screening of {len(events)} events")
        
        # Step 1: Quick screening prompt
        events_text = ""
        for i, event in enumerate(events):
            events_text += f"{i+1}. {event['game_time']}: {event['description']}\n"
        
        screening_prompt = f"""Quickly scan these soccer events and identify which ones could potentially involve shooting actions (shots, goals, saves, blocks, etc.).

Match: {context['home_team']} vs {context['away_team']}

Events:
{events_text}

Return ONLY a JSON array of event numbers that could involve shooting actions:
[1, 3, 7, 12, ...]

Be inclusive - when in doubt, include the event. Only exclude obvious non-shooting events like substitutions, cards, etc."""

        try:
            response = requests.post(
                self.config.base_url,
                headers=self.headers,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": screening_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.0  # Lower temperature for screening
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"Screening API error: {response.status_code}")
                # Fallback: analyze all events
                return self.analyze_batch_single_prompt(events, context)
            
            result = response.json()
            response_text = result['choices'][0]['message']['content']
            
            # Parse screening results
            potential_indices = self._parse_screening_response(response_text)
            self.request_count += 1
            
        except Exception as e:
            print(f"Screening error: {e}")
            # Fallback: analyze all events
            return self.analyze_batch_single_prompt(events, context)
        
        # Step 2: Detailed analysis of selected events
        results = [self._default_response() for _ in events]
        
        if potential_indices:
            selected_events = [events[i-1] for i in potential_indices if 1 <= i <= len(events)]
            print(f"    Step 2 - Detailed analysis of {len(selected_events)} potential shooting events")
            
            if selected_events:
                detailed_results = self.analyze_batch_single_prompt(selected_events, context)
                
                # Map results back to original positions
                for i, event_num in enumerate(potential_indices):
                    if 1 <= event_num <= len(events) and i < len(detailed_results):
                        results[event_num - 1] = detailed_results[i]
        
        print(f"    Two-pass analysis completed")
        return results
    
    def _parse_screening_response(self, response_text: str) -> List[int]:
        """Parse screening response to get list of event numbers"""
        try:
            # Look for JSON array of numbers
            array_match = re.search(r'\[[\d,\s]*\]', response_text)
            if array_match:
                json_array = json.loads(array_match.group(0))
                if isinstance(json_array, list):
                    return [int(x) for x in json_array if str(x).isdigit()]
            
            # Fallback: look for individual numbers
            numbers = re.findall(r'\b\d+\b', response_text)
            return [int(n) for n in numbers if 1 <= int(n) <= 1000]  # Reasonable range
            
        except Exception as e:
            print(f"Screening parse error: {e}")
            return []
    
    # ========================================
    # STRATEGY 5: Caching
    # ========================================
    def analyze_with_cache(self, event: Dict, context: Dict) -> Dict:
        """
        Cache results for identical descriptions
        """
        cache_key = event['description'].lower().strip()
        
        if cache_key in self.cache:
            print(f"    Cache hit: '{cache_key[:30]}...'")
            return self.cache[cache_key].copy()
        
        result = self.analyze_single_event(event, context)
        self.cache[cache_key] = result
        return result
    
    # ========================================
    # Helper Methods
    # ========================================
    def analyze_single_event(self, event: Dict, context: Dict) -> Dict:
        """Analyze single event (used by parallel processing)"""
        prompt = f"""Analyze this soccer event and return ONLY JSON:

Match: {context['home_team']} (home) vs {context['away_team']} (away)
Time: {event['game_time']}
Event: "{event['description']}"

Return: {{
  "is_goal": true/false,
  "is_shooting_action": true/false,
  "is_penalty": true/false,
  "team": "home"/"away"/null,
  "danger_level": integer,  // 1-5 for non-goal shooting actions, 0 otherwise
  "confidence": 0.0-1.0
}}"""

        try:
            response = requests.post(
                self.config.base_url,
                headers=self.headers,
                json={
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": self.config.temperature
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result['choices'][0]['message']['content']
                return self._parse_single_response(response_text)
            else:
                print(f"API error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"    Single event error: {e}")
        
        return self._default_response()
    
    def _parse_single_response(self, response_text: str) -> Dict:
        """Parse single event response"""
        try:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return self._validate_response(parsed)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
        return self._default_response()
    
    def _validate_response(self, parsed: dict) -> Dict:
        """Validate and clean response"""
        defaults = {
            'is_goal': False,
            'is_shooting_action': False,
            'is_penalty': False,
            'team': None,
            'danger_level': 0,
            'confidence': 0.0
        }
        
        for key, default in defaults.items():
            if key not in parsed:
                parsed[key] = default
        
        # Type validation
        for bool_field in ['is_goal', 'is_shooting_action', 'is_penalty']:
            if not isinstance(parsed[bool_field], bool):
                try:
                    parsed[bool_field] = bool(parsed[bool_field])
                except:
                    parsed[bool_field] = str(parsed[bool_field]).lower() in ['true', '1', 'yes']
        
        try:
            parsed['confidence'] = max(0.0, min(1.0, float(parsed['confidence'])))
        except:
            parsed['confidence'] = 0.0
        
        # Validate danger_level
        if 'danger_level' not in parsed:
            parsed['danger_level'] = 0
        try:
            parsed['danger_level'] = max(0, min(5, int(parsed['danger_level'])))
        except:
            parsed['danger_level'] = 0
        
        if parsed['team'] not in ['home', 'away', None]:
            parsed['team'] = None
            
        return parsed
    
    def _default_response(self):
        return {
            'is_goal': False,
            'is_shooting_action': False,
            'is_penalty': False,
            'team': None,
            'danger_level': 0,
            'confidence': 0.0
        }

# ========================================
# Optimized File Processing
# ========================================
def process_file_batch_optimized(input_path, output_path, analyzer, strategy="single_prompt"):
    """
    Process file with chosen batch strategy
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    home_team = data.get('home', {}).get('name', 'Home Team')
    away_team = data.get('away', {}).get('name', 'Away Team')
    
    context = {
        'home_team': home_team,
        'away_team': away_team
    }

    # Prepare events
    annotations = data.get('annotations', [])
    events_to_process = []
    
    for annotation in annotations:
        if annotation.get('description'):
            events_to_process.append({
                'description': annotation['description'],
                'game_time': annotation.get('gameTime', '0 - 0:00'),
                'annotation': annotation  # Keep original for later
            })
    
    if not events_to_process:
        print("  No events to process")
        return
    
    print(f"  Processing {len(events_to_process)} events with strategy: {strategy}")
    
    # Choose processing strategy
    start_time = time.time()
    
    if strategy == "single_prompt":
        # Process in batches using single prompt
        all_results = []
        batch_size = analyzer.config.batch_size
        
        for i in range(0, len(events_to_process), batch_size):
            batch = events_to_process[i:i + batch_size]
            batch_results = analyzer.analyze_batch_single_prompt(batch, context)
            all_results.extend(batch_results)
            
            print(f"    Batch {i//batch_size + 1}: {len(batch)} events -> {len(batch_results)} results")
    
    elif strategy == "parallel":
        all_results = analyzer.analyze_batch_parallel(events_to_process, context)
    
    elif strategy == "conservative_filter":
        all_results = analyzer.analyze_batch_conservative_filter(events_to_process, context)
    
    elif strategy == "two_pass":
        all_results = analyzer.analyze_batch_two_pass(events_to_process, context)
    
    else:  # cache strategy
        all_results = []
        for event in events_to_process:
            result = analyzer.analyze_with_cache(event, context)
            all_results.append(result)
    
    processing_time = time.time() - start_time
    per_event_time = processing_time / len(events_to_process) if events_to_process else 0
    print(f"  Processing took {processing_time:.1f} seconds ({per_event_time:.2f}s per event)")
    
    # Collect goals and dangerous plays
    goals = []
    dangerous_plays = []

    for event, analysis in zip(events_to_process, all_results):
        if analysis['is_goal']:
            # This is a goal
            goals.append({
                'label': "Goal",
                'team': analysis['team'],
                'time_str': event['game_time'],
                'desc': event['description'],
                'confidence': analysis['confidence']
            })
        elif analysis['danger_level'] > 0:  # Non-goal dangerous play
            dangerous_plays.append({
                'label': "Dangerous Play",
                'team': analysis['team'],
                'time_str': event['game_time'],
                'desc': event['description'],
                'danger_level': analysis['danger_level'],
                'confidence': analysis['confidence']
            })

    # Select top 4 dangerous plays (sort by danger_level descending)
    dangerous_plays.sort(key=lambda x: x['danger_level'], reverse=True)
    top_dangerous = dangerous_plays[:4]

    # Combine goals + top dangerous plays
    events_data = goals + top_dangerous

    print(f"  Found {len(events_data)} key events ({len(goals)} goals + {len(top_dangerous)} dangerous plays)")
    
    # Write results
    output_lines = []
    for e in events_data:
        team_str = e['team'] if e['team'] in ['home', 'away'] else 'unknown'
        conf_str = f"(confidence: {e['confidence']:.2f})"
        
        if e['label'] == "Goal":
            output_lines.append(f"Goal ({team_str}) {e['time_str']}: {e['desc']} {conf_str}")
        else:
            output_lines.append(f"Dangerous Play ({team_str}) {e['time_str']}: {e['desc']} [Danger: {e['danger_level']}/5] {conf_str}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

def main():
    """Main function with batch processing options"""
    print("SoccerNet Analysis with DeepSeek - LLM-Focused Strategies")
    print("=" * 55)
    
    # Configuration
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not API_KEY:
        print("DEEPSEEK_API_KEY environment variable not set!")
        print("Please set your API key: export DEEPSEEK_API_KEY='your_key_here'")
        return
    
    # Choose your strategy:
    # "single_prompt" - Most efficient, batch multiple events per API call
    # "parallel" - Process events in parallel threads
    # "conservative_filter" - Only filter obvious non-game events, LLM analyzes rest
    # "two_pass" - LLM screening then detailed analysis (most thorough)
    # "cache" - Cache identical event descriptions
    
    STRATEGY = "single_prompt"  # Recommended for most cases
    BATCH_SIZE = 8  # Increased batch size for efficiency
    NUM_WORKERS = 3  # For parallel strategy
    
    config = DeepSeekConfig(
        api_key=API_KEY,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    analyzer = BatchDeepSeekAnalyzer(config)
    print(f"Using strategy: {STRATEGY}")
    print(f"Strategy description:")
    
    strategy_descriptions = {
        "single_prompt": "Batch multiple events in single API calls - most efficient",
        "parallel": "Process events in parallel threads - good for rate limit handling",
        "conservative_filter": "LLM analyzes all game events, filters only obvious admin events",
        "two_pass": "LLM screening + detailed analysis - most thorough but slower",
        "cache": "Cache identical event descriptions - good for repeated matches"
    }
    
    print(f"   {strategy_descriptions.get(STRATEGY, 'Custom strategy')}")
    
    # Paths
    input_dir = r"E:\FrankGuo\SoccerNet\llmtest\caption2024"
    output_dir = rf"E:\FrankGuo\SoccerNet\llmtest\deepseek_improved_{STRATEGY}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Found {len(json_files)} files")
    
    total_start = time.time()
    
    for i, filename in enumerate(json_files):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"deepseek_output_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\nProcessing {filename} ({i+1}/{len(json_files)})")
        
        try:
            process_file_batch_optimized(input_path, output_path, analyzer, STRATEGY)
            print(f"Completed: {filename}")
        except Exception as e:
            print(f"Error: {e}")
    
    total_time = time.time() - total_start
    print(f"\nAll files processed in {total_time:.1f} seconds!")
    print(f"Stats: {analyzer.request_count} total API calls")
    print(f"Cache size: {len(analyzer.cache)} entries")
    
    # Strategy recommendations
    print(f"\nStrategy Performance Notes:")
    print(f"   • single_prompt: Best balance of speed and accuracy")
    print(f"   • two_pass: Most thorough but uses 2x API calls")
    print(f"   • conservative_filter: Only filters obvious non-game events")
    print(f"   • parallel: Good for handling rate limits")

if __name__ == "__main__":
    main()