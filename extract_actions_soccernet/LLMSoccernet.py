# Batch Processing Optimized SoccerNet Analysis with DeepSeek API
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
    Optimized DeepSeek analyzer with multiple batch processing strategies
    """
    def __init__(self, config: DeepSeekConfig):
        self.config = config
        self.request_count = 0
        self.cache = {}  # Simple cache for repeated descriptions
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"🚀 DeepSeek analyzer initialized:")
        print(f"   Model: {config.model}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Workers: {config.num_workers}")
    
    # ========================================
    # STRATEGY 1: Multi-Event Single Prompt
    # ========================================
    def analyze_batch_single_prompt(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Analyze multiple events in a single prompt - MOST EFFICIENT
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
- confidence: 0.0-1.0 how certain you are

Return ONLY a JSON array with {len(events)} objects like this:
[
  {{"is_goal": boolean, "is_shooting_action": boolean, "is_penalty": boolean, "team": "home"/"away"/null, "confidence": float}},
  ... // {len(events)} objects total
]"""

        try:
            print(f"    🔄 Batch analyzing {len(events)} events...")
            
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
            
            print(f"    ✅ Batch completed: {len(parsed_results)} results")
            return parsed_results
            
        except Exception as e:
            print(f"    ❌ Batch error: {e}")
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
            print(f"    ❌ Batch parse error: {e}")
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
        
        print(f"    🔄 Parallel processing {len(events)} events with {self.config.num_workers} workers...")
        
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
        
        print(f"    ✅ Parallel processing completed: {len(results)} results")
        return results
    
    # ========================================
    # STRATEGY 3: Pre-filtering + Batch
    # ========================================
    def analyze_batch_smart_filter(self, events: List[Dict], context: Dict) -> List[Dict]:
        """
        Pre-filter obvious non-shooting events, then batch process the rest
        """
        # Quick rule-based pre-filtering
        shooting_events = []
        results = []
        
        non_shot_keywords = [
            'substitution', 'yellow card', 'red card', 'offside', 'throw-in',
            'corner kick preparation', 'free kick preparation', 'pass', 
            'cross', 'dribble', 'tackle', 'clearance', 'foul', 'injury'
        ]
        
        shot_keywords = [
            'shot', 'shoot', 'strike', 'header', 'volley', 'effort', 
            'finish', 'attempt', 'goal', 'score', 'save', 'block', 'miss'
        ]
        
        for i, event in enumerate(events):
            desc_lower = event['description'].lower()
            
            # Quick non-shot detection
            if any(keyword in desc_lower for keyword in non_shot_keywords):
                results.append((i, {
                    'is_goal': False,
                    'is_shooting_action': False,
                    'is_penalty': False,
                    'team': None,
                    'confidence': 0.95  # High confidence in non-shot
                }))
                continue
            
            # Quick shot detection
            if any(keyword in desc_lower for keyword in shot_keywords):
                shooting_events.append((i, event))
            else:
                # Uncertain - send to LLM
                shooting_events.append((i, event))
        
        print(f"    🎯 Pre-filtered: {len(shooting_events)} potential shooting events out of {len(events)}")
        
        # Batch process remaining events
        if shooting_events:
            indices, events_to_process = zip(*shooting_events)
            llm_results = self.analyze_batch_single_prompt(list(events_to_process), context)
            
            for idx, result in zip(indices, llm_results):
                results.append((idx, result))
        
        # Sort results by original order and return
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    # ========================================
    # STRATEGY 4: Caching
    # ========================================
    def analyze_with_cache(self, event: Dict, context: Dict) -> Dict:
        """
        Cache results for identical descriptions
        """
        cache_key = event['description'].lower().strip()
        
        if cache_key in self.cache:
            print(f"    💾 Cache hit: '{cache_key[:30]}...'")
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
            print(f"    ❌ Single event error: {e}")
        
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
        
        if parsed['team'] not in ['home', 'away', None]:
            parsed['team'] = None
            
        return parsed
    
    def _default_response(self):
        return {
            'is_goal': False,
            'is_shooting_action': False,
            'is_penalty': False,
            'team': None,
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
        print("  ⚠️ No events to process")
        return
    
    print(f"  📊 Processing {len(events_to_process)} events with strategy: {strategy}")
    
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
    
    elif strategy == "smart_filter":
        all_results = analyzer.analyze_batch_smart_filter(events_to_process, context)
    
    else:  # cache strategy
        all_results = []
        for event in events_to_process:
            result = analyzer.analyze_with_cache(event, context)
            all_results.append(result)
    
    processing_time = time.time() - start_time
    per_event_time = processing_time / len(events_to_process) if events_to_process else 0
    print(f"  ⏱️ Processing took {processing_time:.1f} seconds ({per_event_time:.2f}s per event)")
    
    # Convert results to your existing format
    events_data = []
    for i, (event, analysis) in enumerate(zip(events_to_process, all_results)):
        if not analysis['is_shooting_action']:
            continue
        
        label = "Goal" if analysis['is_goal'] else "No Goal"
        if analysis['is_goal'] and analysis['is_penalty']:
            label = "Penalty"
        
        events_data.append({
            'label': label,
            'time_str': event['game_time'],
            'desc': event['description'],
            'confidence': analysis['confidence']
        })
    
    print(f"  ⚽ Found {len(events_data)} shooting events")
    
    # Write results
    output_lines = []
    for e in events_data:
        confidence_str = f" (confidence: {e['confidence']:.2f})"
        output_lines.append(f"{e['label']} {e['time_str']} {e['desc']}{confidence_str}")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

def main():
    """Main function with batch processing options"""
    print("🚀 Batch Processing SoccerNet Analysis with DeepSeek")
    print("=" * 50)
    
    # Configuration
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    if not API_KEY:
        print("❌ DEEPSEEK_API_KEY environment variable not set!")
        print("Please set your API key: export DEEPSEEK_API_KEY='your_key_here'")
        return
    
    STRATEGY = "single_prompt"  # Choose: "single_prompt", "parallel", "smart_filter", "cache"
    BATCH_SIZE = 5  # Events per batch (for single_prompt strategy)
    NUM_WORKERS = 3  # Parallel workers (for parallel strategy)
    
    config = DeepSeekConfig(
        api_key=API_KEY,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    analyzer = BatchDeepSeekAnalyzer(config)
    print(f"📋 Using strategy: {STRATEGY}")
    
    # Paths
    input_dir = r"E:\FrankGuo\SoccerNet\llmtest\caption2024"
    output_dir = rf"E:\FrankGuo\SoccerNet\llmtest\deepseek_batch_{STRATEGY}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"📁 Found {len(json_files)} files")
    
    total_start = time.time()
    
    for i, filename in enumerate(json_files):
        input_path = os.path.join(input_dir, filename)
        output_filename = f"deepseek_output_{os.path.splitext(filename)[0]}.txt"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n🏈 Processing {filename} ({i+1}/{len(json_files)})")
        
        try:
            process_file_batch_optimized(input_path, output_path, analyzer, STRATEGY)
            print(f"✅ Completed: {filename}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    total_time = time.time() - total_start
    print(f"\n🎉 All files processed in {total_time:.1f} seconds!")
    print(f"📊 Stats: {analyzer.request_count} total API calls")
    print(f"💾 Cache size: {len(analyzer.cache)} entries")

if __name__ == "__main__":
    main()