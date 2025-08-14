import ollama
import json
import time

# === INPUT: Your dataset of soccer commentaries ===
commentary_examples = [
    {
        "id": 1,
        "goal": True,
        "text": "Goal! Andre Ayew (Swansea) was in the right place at the right time to get to the rebound inside the box and gleefully rifles the ball into the right side of the goal. It's 1:1."
            "Been key to them over the last couple of years. Here's Montero."
            "Gomes. Great save."
            "Ayoub. Oh, wonderful stop."
            "Ayoub again. 1-1."
            "Debut goal for Andre Ayoub"
            "and the champions are pegged back"
            "at home."
            "The Welsh invaders with reason to smile."
            "Reason given them by their"
            "Garnier new boy."
            "Courtois did all he could."
            "It wasn't enough. Well, just as I"
            "thought, Swansea were on the back foot."
            "What a fantastic piece of play."
            "Great play. Great ball into the box."
            "Really good head-on. Fabulous save."
            "And I thought they'd chanced Gomes."
            "Gary Cahill makes that stop."
            "Helps out his goalkeeper with a fantastic"
            "block. Throwing his body in the way."
            "But then Ayoub, as quick as anybody,"
            "gets up, reacts, but then"
            "keeps his composure."
            "Wait here. He just has a little pause."
            "He stops, drags it back"
            "and fires it into the far corner."
            "Fantastic finish from Ayoub."
    },
    {
        "id": 2,
        "goal": True,
        "text": "Goal! Eden Hazard provides Branislav Ivanovic (Chelsea) with a nice pass inside the box. It allows him to finish with a precise effort into the bottom right corner. 1:0."
            "Play again for Hazard."
            "Already inside the area."
            "What a fantastic maneuver by the Belgian."
            "Ivanovic."
            "Ivanovic scores again."
            "Well, Ivanovic scores again."
            "Although the player is from Eden Hazard."
            "Minute 14 of the first half."
            "Perhaps Barley is too soft in defense."
            "But great individual action from Hazard."
            "Ivanovic once again faithful to his appointment with the goal."
            "Score the first for Chelsea 1-0."
            "Yes, the Barley is going to be soft here."
            "With that excessively easy turnover."
            "Then Hazard's thing is impressive."
            "Hazard's action is..."
            "Well, it's pure Hazard."
            "The individual action, the start, the different changes of speed, the breaks."
            "And then the perfect pass because Ivanovic sneaks in."
            "No one marks him."
            "And Hazard, with the eye in the back of his head, sees him perfectly."
            "And he assists him."
            "Being totally busy and busy with business."
            "It seems incredible that Hazard could have the coldness and vision to know where his teammate was."
            "And not only did he know it, but he put in a perfect ball."
    },
    {
        "id": 3,
        "goal": False,
        "text": "Branislav Ivanovic (Chelsea) takes a first-time shot from the edge of the box, but his effort is well blocked by the defender."
            "Diego Costa anticipates."
            "Jones can score."
            "It ends up opening square."
            "He asks for it back."
            "Ivanovic."
            "Ivanovic coming again."
            "They asked for a hand."
            "He does it clearly and ostensibly."
            "The Serbian side."
            "Who complains bitterly that this decision did not end with the 11 meters."
            "I honestly couldn't appreciate it live."
            " Me neither."
            "Let's watch some replay."
            "With that square ball back for Branislav Ivanovic's shot."

    },
    # Add more commentary entries as needed...
]

examples = """

Commentary:
Goal! Andre Ayew reacts quickly to a rebound and fires the ball into the bottom right corner.

Q1: What happened in the play?
A1: Ayew capitalized on a rebound and scored with a composed finish.

Q2: What are the key moments that led to that outcome?
A2: The initial shot forced a save, the ball rebounded into the box, and Ayew reacted faster than the defenders.

"""

# === FUNCTION: Generate Q&A from commentary ===
def generate_qa_from_commentary(commentary_text, final_action_bool):
    if final_action_bool:
        final_action = 'Goal'
    else:
        final_action = 'close chance'


    prompt = f"""
You are analyzing a soccer play that ended in a {final_action}. Based on the commentaries provided, answer the following:

1. What happened in the play? 
2. What are the key moments that lead to that this {final_action}? 


Commentary:
{commentary_text}

Questions and Answers:
"""
    response = ollama.chat(
        model='llama3:instruct',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['message']['content'].strip()

# === PROCESS: Loop over all commentaries ===
results = []

for example in commentary_examples:
    try:
        print(f"Processing commentary ID {example['id']}...")
        qa_output = generate_qa_from_commentary(example['text'], example['goal'])
        
        results.append({
            "id": example["id"],
            "commentary": example["text"],
            "qa": qa_output
        })

        time.sleep(1)  # Respect model rate
    except Exception as e:
        print(f"Error processing ID {example['id']}: {e}")

# === OUTPUT: Save to JSON ===
with open("soccer_qa_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Q&A generation complete. Results saved to soccer_qa_output.json")
