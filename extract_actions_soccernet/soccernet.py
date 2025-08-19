#modified version for soccernet
import os
import json
import re
from score_chance_play import ScoreChancePlay
import pandas as pd
import hashlib
import csv
import math
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
    #'corner kick', 'free kick', we want to keep actions following a corner kick or free kick

    # Non-shot actions - defensive, passing, clearing, possession
    'cleared after', 'attempted to dribble', 'attempts to send a pass',
    'effort is blocked', 'blocked the effort', 'earned a corner',
    'held onto the ball', 'holds the ball', 'both sides enjoying spells',
    'played long-ball football', 'attempted to hold onto the ball',
    'no time for the supporters', 'was full of action', 'thrilling moments',

    # Add your recent phrases here:
    'send over a cross',
    'fails to send a pass',
    'tries to send a pass',
    #'is blocked',
    #'pass ends up', didnt find
    'passes end up',
    'signal a throw-in',
    #'out of play',
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
    #if half == '2':
        #total_seconds += 45 * 60
    return int(half), total_seconds 

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


def exclude_irrelevant_phrases(description):
    desc = description.lower()
    # Skip irrelevant/statistical/set-piece prep
    if EXCLUSION_REGEX.search(desc):
        return 
    

def is_shooting_action(description, irrelevant_phrases_bool):
    """Check if description is a genuine shot attempt (goal or no goal)."""
    desc = description.lower()
    # Must contain a shooting term
    if irrelevant_phrases_bool:
        if EXCLUSION_REGEX.search(desc):
            return False

    if not SHOOTING_REGEX.search(desc):
        return False
    # Saves/blocks/denied remain valid as shot attempts
    return True

def cluster_events(events, time_gap=60.0):
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
                #if current_cluster[-1]['half_time'] == e['half_time']:
                prev_time = current_cluster[-1]['time'] + e['half_time']*2700 - 2700
                if (e['time']+ e['half_time']*2700 - 2700) - prev_time <= time_gap:
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




def is_goal_event(annotation, goal_events, home_team, away_team, test_goal_kw, test_event_time):
    description = annotation['description'].lower()
    total_minutes = parse_game_time(annotation['gameTime']) ##changed this functions, this might not work, gibran
    if total_minutes is None:
        return None, False  # Added penalty flag

    penalty_detected = 'penalty' in description
    

    
    #TESTED THIS CHECKS, THEY ARE RETURNING NON-GOAL PLAYS AS GOALS, THIS SHOULD BE REMOVED, Gibran
    #delete consecutive events
    # if test_event_time:
    #     for event in goal_events:
    #         if 0 <= (total_minutes - event['minute']) <= 1.5:
    #             last_name_lower = event['last_name'].lower()
    #             full_name_lower = event['full_name'].lower()
    #             if last_name_lower in description or full_name_lower in description:
    #                 return True, event['team'], penalty_detected  # Return penalty status

    # if(test_goal_kw):
    #     goal_keywords = ['goal!', 'scores!', 'netted!', 'converts!', 'finishes!',
    #                     'goal:', 'scores :', 'scored!', 'goal -', 'scores -']
    #     if any(kw in description for kw in goal_keywords):
    #         if home_team.lower() in description:
    #             return True, 'home', penalty_detected
    #         elif away_team.lower() in description:
    #             return True, 'away', penalty_detected

    #
    # if OUTCOME_REGEX.search(description):
    #     #if 'saved' not in description and 'blocked' not in description and 'clear' not in description: ##FOUND THIS TO BE IRRELEVANT, so i comment it, Gibran
    #     if home_team.lower() in description:
    #         return True, 'home', penalty_detected
    #     elif away_team.lower() in description:
    #         return True, 'away', penalty_detected

    return None, penalty_detected


def extract_match_info(file_path: str):
    # normalize separators (Windows/Linux safe)
    parts = os.path.normpath(file_path).split(os.sep)
    # last parts before the JSON file
    league = parts[-4]  
    season = parts[-3]   
    match = parts[-2]#.replace("Labels-caption.json", "").strip(os.sep)  

    return league, season, match
    


# -----------------------------
# File processing
# -----------------------------
def process_file(input_path, test_goal_kw = True, irrelevant_phrases_bool = True, test_event_time=True):
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

    league, season, match = extract_match_info(input_path)

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
            team, is_penalty = is_goal_event(annotation, goal_events, home_team, away_team, test_goal_kw, test_event_time)
        elif is_shooting_action(desc, irrelevant_phrases_bool):
            is_goal =False 
            team,_ = is_goal_event(annotation, goal_events, home_team, away_team, test_goal_kw, test_event_time)
        else:
            continue

        # Determine label based on penalty status
        if is_goal:
            label = "Penalty" if is_penalty else "Goal"
        else:
            label = "No Goal"
            
        half, time_seconds = parse_game_time(annotation['gameTime']) #changed from min to seconds, gibran
         # exclude no goals that happen after 44 minutes, 45+ min footage is only included when there is  agoal
        if time_seconds > 2640 and label == 'No Goal':
            continue

        events_data.append({
            'league': league,
            'season': season,
            'match': match,
            'label': label,
            'minute': time_seconds/60,
            'time': time_seconds,
            'half_time': half,
            'time_str': annotation['gameTime'],
            'caption': desc,

        })

    clustered_events = cluster_events(events_data, time_gap=60.0)

    return clustered_events


def make_caption_id(file_path, caption):
    # Extract the match name (last part of path)
    match_name = os.path.basename(file_path).replace(" ", "_")
    # Normalize caption text for consistency
    norm_caption = caption.strip().lower()
    # Create a short hash of the caption
    hash_str = hashlib.md5(norm_caption.encode("utf-8")).hexdigest()[:10]
    # Combine match name + hash
    #return f"{match_name}_{hash_str}"
    return hash_str


def output_lines_file(events, output_path, threshold, close_chances_per_game):
    output_lines = []
    score = 0
    goals = 0
    close_chances = 0
    penalty =0
    all_captions = []
    accepted_events = []
    #print(f'file threshold: {threshold}')
    for e in events:
        if e['score'] >= threshold: 
            if e['label'] == 'No Goal' and close_chances >= close_chances_per_game:
                continue
            line = f"{e['label']} {e['time_str']} [{e['score']}] {e['caption']}"
            output_lines.append(line)
            e['thold'] =threshold
            e['hash'] = make_caption_id(output_path, e['caption'])
            

            if e['label'] == 'Goal':
                goals +=1
            elif e['label'] == 'No Goal':
                score += e['score']
                close_chances +=1 
            elif e['label'] == 'Penalty':
                penalty +=1
                continue
            
            all_captions.append([e['hash'], line])
            accepted_events.append(e)
            
    
    print(close_chances)
    #output_lines = [f"{e['label']} {e['time_str']} {e['desc']}" for e in events]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.write("\n".join(output_lines))

    return score, goals, close_chances, penalty, all_captions, accepted_events


def get_score_threshold(events, accepted_amt):
    """
    Calculates danger score value for all non goal desccriptions and 
    return the threshold point at the nth caption
    Args:
        events: all events for that captions file
        accepted_amt: number of close chance actions kept
    Returns:
    scores and events
        threshold acceptance score
    """
    scorer = ScoreChancePlay()
    scorer.include_location_score = False # Not include location score to maintain variability of shot distances
    close_chance_scores = []
    close_chance_score = 0
    j = 0
    for event in events:
        if event['label'] == 'No Goal':
            score = scorer.calculate_danger_score(event['caption'])
            event['score'] = score
            close_chance_scores.append([score, event]) 
            close_chance_score = close_chance_score + score
            j+=1
        else:
            event['score'] = 15 # GIVE MAXIMUM SCORE TO GOALS
        
    if len(close_chance_scores) > accepted_amt:
        #close_chance_scores.sort(key=lambda x: x[0], reverse=True)
        #return events, close_chance_scores[accepted_amt][0] #return score of the threshold
        #its better to return avg
        return events, close_chance_score/j
    else:
        return events, -2 #lowest score
    

def score_analysis(events, threshold):
    total_score = 0
    for e in events:
        if e['score'] >threshold: 
            total_score += e['score']
    print(f'total danger score = {total_score}')


# -----------------------------
# Main entry
# -----------------------------
def get_all_captions(input_dir, output_dir, test_goal_kw = False, irrelevant_phrases_bool =True, test_event_time=True, close_chances_per_game = 100):
    total_goals = 0
    total_non_goals = 0
    total_score = 0
    total_penalty = 0
    number_events = 0
    all_captions =[]
    all_events = []
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.json'):
                input_path = os.path.join(root, filename)
                #print(input_path)
                parts = root.split("\\")
                output_filename = f'output_{parts[5]}_{parts[6]}_{parts[7]}.txt'
                #output_filename = f"output_{os.path.splitext(root)[1]}.txt"
                output_path = os.path.join(output_dir, output_filename)
                try:
                    events = process_file(input_path,test_goal_kw, irrelevant_phrases_bool, test_event_time)
                    events, danger_score_threshold = get_score_threshold(events, close_chances_per_game)
                    number_events += len(events)
                    score, goals, non_goals, penalty, captions, events= output_lines_file(events, output_path, danger_score_threshold, close_chances_per_game) #pass -5 as threshold to output all
                    total_score += score
                    total_goals += goals
                    total_non_goals += non_goals
                    total_penalty += penalty
                    all_captions.extend(captions)
                    all_events.extend(events)
                    #print(f"Processed: {filename}")
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
    print(f'total danger score = {total_score}')
    print(f'num events = {number_events}')
    print(f'avg = {total_score/total_non_goals}')
    print(f'Goals: {total_goals}, No goals: {total_non_goals}, Penalties: {total_penalty}, Total= {total_goals+total_non_goals+total_penalty}')

    return all_events
    # with open("captions.csv", "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "caption"])  # header
    #     writer.writerows(all_captions)

if __name__ == "__main__":
    output_dir = r"results\\extract_actions_soccernet\\goal_results_2024"
    input_dir = r"D:\\soccernetCaptionDataset\\caption-2024"
    # Testing code, to see how different kw conditions affected the result
    #df_goalkw_true = pd.DataFrame(get_all_captions(output_dir, test_goal_kw=False, irrelevant_phrases_bool=True, test_event_time=False), columns= ['id', 'caption'])
    #df_goalkw_false = pd.DataFrame(get_all_captions(output_dir, test_goal_kw=False, irrelevant_phrases_bool=False, test_event_time=False), columns= ['id', 'caption'])
    #df_diff= df_goalkw_false[~df_goalkw_false['id'].isin(df_goalkw_true['id'])]
    #df_diff.to_csv('diff.csv', index=False)

    
    all_events = get_all_captions(input_dir, output_dir, test_goal_kw=False, irrelevant_phrases_bool=True, test_event_time=False, close_chances_per_game= 4)
    all_actions_id = pd.DataFrame( [ [e['hash'], e['caption']] for e in all_events ], columns= ['id', 'caption'])
    all_actions_id.to_csv('results/extract_actions_soccernet/all_actions_id.csv', index=False)
    
    with open('results/extract_actions_soccernet/all_events.json', mode= 'w', encoding='utf-8') as f:
        json.dump(all_events, f, indent=2)

        ##get in json, players(both names and id), teams, league and season, mathc, ht

    
    



