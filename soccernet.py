#modified version for soccernet
import os
import json
import re

# -----------------------------
# Outcome phrases classification
# -----------------------------
UNAMBIGUOUS_OUTCOME_PHRASES = [
    'ended up inside', 'found the net', 'back of the net',
    'bottom corner', 'top corner', 'into the goal', 'past the keeper',
    'beaten the goalkeeper', 'beaten the keeper', 'into the net',
    'into the top', 'into the bottom', 'into the left', 'into the right',
    'deflecting in', 'tapping the ball in', 'empty net'
]
AMBIGUOUS_OUTCOME_PHRASES = [
    r'plant\s+.*?\s+header',
    r'chipped\s+the\s+ball'
]
GOAL_KEYWORDS = r'goal|score[sd]?|net(?:ted)?'

unambiguous_outcome_str = '|'.join([re.escape(p) for p in UNAMBIGUOUS_OUTCOME_PHRASES])
ambiguous_outcome_str = '|'.join(AMBIGUOUS_OUTCOME_PHRASES)

OUTCOME_REGEX = re.compile(
    rf"""
    (
        (?:{unambiguous_outcome_str})
    )
    |
    (
        (?:
            (?:{ambiguous_outcome_str})
            (?:\W+\w+){{0,20}}?
            \W+(?:
                (?:{unambiguous_outcome_str})
                |
                \b(?:{GOAL_KEYWORDS})\b
            )
        )
        |
        (?:
            (?:
                (?:{unambiguous_outcome_str})
                |
                \b(?:{GOAL_KEYWORDS})\b
            )
            (?:\W+\w+){{0,20}}?
            \W+(?:{ambiguous_outcome_str})
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL
)

# -----------------------------
# Shooting action keywords
# -----------------------------
SHOOTING_TERMS = [
    'shot', 'shoot', 'strike', 'curl', 'drive', 'fire',
    'volley', 'header', 'effort', 'finish',
    'blast', 'hit', 'attempted', 'took a shot', 'with a shot',
    'tries', 'strikes', 'howitzer', 'lob', 'chip', 'tap in',
    'precisely into', 'plant a header', 'finishes with',
    'chipped the ball', 'tapping the ball', 'rifles', 'pulls the trigger',
    'unleashes', 'ripped', 'slammed', 'blazes', 'crashes', 'meets'
]
SHOOTING_PATTERN = r'|'.join([r'\b' + re.escape(term) + r'\b' for term in SHOOTING_TERMS])
SHOOTING_REGEX = re.compile(SHOOTING_PATTERN, re.IGNORECASE)

# -----------------------------
# Refined exclusion list
# -----------------------------

EXCLUSION_PHRASES = [
    # Stats and generic commentary
    'total number', 'attempts is', 'shots are', 'statistics', 'ratio',
    'number of', 'count is', 'figures are', 'tally is', 'currently',
    'we can have a look', 'statistics now', 'attendance is',

    # Set-piece prep without shot
    'prepares to take', 'take a corner', 'swings in a corner',
    'corner kick', 'free kick',

    # Non-shot actions - defensive, passing, clearing, possession
    'cleared after', 'attempted to dribble', 'attempts to send a pass',
    'effort is blocked', 'blocked the effort', 'earned a corner',
    'held onto the ball', 'holds the ball', 'both sides enjoying spells',
    'played long-ball football', 'attempted to hold onto the ball',
    'no time for the supporters', 'was full of action', 'thrilling moments',

    # Add your recent phrases here:
    'send over a cross',
    'send a pass',
    'is blocked',
    'pass ends up',
    'passes end up',
    'signal a throw-in',
    'out of play',
    'throws a throw-in',
    'possession',
    'short passes'
]

# Build regex pattern for exclusion phrases, escaping properly and allowing substring matches for multiword phrases
EXCLUSION_REGEX = re.compile(
    r'(' + '|'.join([re.escape(p) for p in EXCLUSION_PHRASES]) + r')',
    re.IGNORECASE
)

# -----------------------------
# Goal event extraction
# -----------------------------
def extract_goal_events(data):
    goal_events = []
    for team in ['home', 'away']:
        for player in data['lineup'][team]['players']:
            for fact in player['facts']:
                if fact['type'] == '3':  # Goal
                    match = re.search(r'(\d+)\'', fact['time'])
                    if match:
                        minute = int(match.group(1))
                        last_name = player['name'].split()[0]
                        full_name = player.get('long_name', player['name'])
                        goal_events.append({
                            'minute': minute,
                            'last_name': last_name,
                            'full_name': full_name,
                            'team': team
                        })
    return goal_events

# -----------------------------
# Time parsing helpers
# -----------------------------
def parse_game_time(game_time):
    parts = game_time.split(' - ')
    if len(parts) != 2:
        return None
    half, time_str = parts
    time_parts = time_str.split(':')
    if not time_parts[0].isdigit() or not time_parts[1].isdigit():
        return None
    minutes = int(time_parts[0])
    seconds = int(time_parts[1])
    total_seconds = minutes * 60 + seconds
    if half == '2':
        total_seconds += 45 * 60
    return total_seconds / 60.0

def parse_game_time_for_sorting(game_time):
    parts = game_time.split(' - ')
    if len(parts) != 2:
        return (0, 0, 0)
    half, time_str = parts
    time_parts = time_str.split(':')
    if len(time_parts) < 2:
        time_parts.append('00')
    minute = int(time_parts[0]) if time_parts[0].isdigit() else 0
    second = int(time_parts[1]) if time_parts[1].isdigit() else 0
    half = 1 if half == '1' else 2
    return (half, minute, second)

# -----------------------------
# Event classification
# -----------------------------
def is_goal_event(annotation, goal_events, home_team, away_team):
    description = annotation['description'].lower()
    total_minutes = parse_game_time(annotation['gameTime'])
    if total_minutes is None:
        return False, None, False  # Added penalty flag

    penalty_detected = 'penalty' in description
    
    for event in goal_events:
        if 0 <= (total_minutes - event['minute']) <= 1.5:
            last_name_lower = event['last_name'].lower()
            full_name_lower = event['full_name'].lower()
            if last_name_lower in description or full_name_lower in description:
                return True, event['team'], penalty_detected  # Return penalty status

    goal_keywords = ['goal!', 'scores!', 'netted!', 'converts!', 'finishes!',
                    'goal:', 'scores :', 'scored!', 'goal -', 'scores -']
    if any(kw in description for kw in goal_keywords):
        if home_team.lower() in description:
            return True, 'home', penalty_detected
        elif away_team.lower() in description:
            return True, 'away', penalty_detected

    if OUTCOME_REGEX.search(description):
        if 'saved' not in description and 'blocked' not in description and 'clear' not in description:
            if home_team.lower() in description:
                return True, 'home', penalty_detected
            elif away_team.lower() in description:
                return True, 'away', penalty_detected

    return False, None, penalty_detected

    
def is_shooting_action(description):
    """Check if description is a genuine shot attempt (goal or no goal)."""
    desc = description.lower()
    # Skip irrelevant/statistical/set-piece prep
    if EXCLUSION_REGEX.search(desc):
        return False
    # Must contain a shooting term
    if not SHOOTING_REGEX.search(desc):
        return False
    # Saves/blocks/denied remain valid as shot attempts
    return True

def cluster_events(events, time_gap=1.0):
    """Cluster non-goal events while preserving all goals and penalties"""
    clustered = []
    current_cluster = []
    
    for e in events:
        # Preserve all goals and penalties individually
        if e['label'] in ["Goal", "Penalty"]:
            if current_cluster:
                clustered.append(select_event(current_cluster))
                current_cluster = []
            clustered.append(e)
        else:
            # Cluster non-goal events
            if not current_cluster:
                current_cluster.append(e)
            else:
                prev_time = current_cluster[-1]['minute']
                if e['minute'] - prev_time <= time_gap:
                    current_cluster.append(e)
                else:
                    clustered.append(select_event(current_cluster))
                    current_cluster = [e]
    
    # Handle remaining non-goal events
    if current_cluster:
        clustered.append(select_event(current_cluster))
        
    return clustered

def select_event(cluster):
    """For non-goal clusters, keep the last event"""
    return cluster[-1]  # Always last event in cluster
# -----------------------------
# File processing
# -----------------------------
def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    home_team = data.get('home', {}).get('name', 'Home Team')
    away_team = data.get('away', {}).get('name', 'Away Team')
    goal_events = extract_goal_events(data)

    events_data = []

    sorted_annotations = sorted(
        data['annotations'],
        key=lambda x: parse_game_time_for_sorting(x['gameTime'])
    )

    for annotation in sorted_annotations:
        if not annotation.get('description'):
            continue

        desc = annotation['description']
        is_goal = False
        team = None
        is_penalty = False

        # Handle goal annotations
        if annotation.get('label') in ['soccer-ball', 'soccer-ball-own']:
            is_goal=True
            _, team, is_penalty = is_goal_event(annotation, goal_events, home_team, away_team)
        elif is_shooting_action(desc):
            is_goal, team,_ = is_goal_event(annotation, goal_events, home_team, away_team)
        else:
            continue

        # Determine label based on penalty status
        if is_goal:
            label = "Penalty" if is_penalty else "Goal"
        else:
            label = "No Goal"
            
        minute = parse_game_time(annotation['gameTime'])
        events_data.append({
            'label': label,
            'minute': minute,
            'time_str': annotation['gameTime'],
            'desc': desc
        })

    clustered_events = cluster_events(events_data, time_gap=1.0)

    output_lines = [f"{e['label']} {e['time_str']} {e['desc']}" for e in clustered_events]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

# -----------------------------
# Main entry
# -----------------------------
def main():
    input_dir = r"E:\FrankGuo\SoccerNet\all_captions_2024"
    output_dir = r"E:\FrankGuo\SoccerNet\goal_results_2024"
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"output_{os.path.splitext(filename)[0]}.txt"
            output_path = os.path.join(output_dir, output_filename)
            try:
                process_file(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
