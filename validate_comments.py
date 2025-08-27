import ollama
import json
import time
import os
import pandas as pd

example = """

Example:

Input:
full_caption: "Branislav Ivanovic (Chelsea) takes a first-time shot from the edge of the box, but his effort is well blocked by the defender."
echoes: "[1862.04, 1863.04], Chelsea had a peak of intensity.; [1864.04, 1865.04], The one where he scored the goal.; [1867.04, 1869.04], He created a couple of very clear chances.; [1870.04, 1876.04], But he found it so easy again that the team has returned to another small valley in terms of intensity.; [1877.04, 1878.04], Diego Costa anticipates.; [1879.04, 1880.04], Jones can score.; [1881.04, 1882.04], It ends up opening square.; [1883.04, 1884.04], He asks for it back.; [1884.04, 1885.04], Ivanovic.; [1886.04, 1887.04], Ivanovic coming again.; [1888.04, 1889.04], They asked for a hand.; [1890.04, 1891.04], He does it clearly and ostensibly.; [1892.04, 1893.04], The Serbian side.; [1894.04, 1898.04], Who complains bitterly that this decision did not end with the 11 meters.; [1899.04, 1901.04], I honestly couldnt appreciate it live.; [1902.04, 1903.04], Me neither.; [1904.04, 1905.04], Let's watch some replay.; [1906.04, 1910.04], With that square ball back for Branislav Ivanovic's shot.; [1911.04, 1912.04], He was jumping now, King.; [1912.04, 1913.04], He was jumping now, King.; [1914.04, 1915.04], Jones was playing.; [1916.04, 1917.04], He tries to control her.; [1918.04, 1921.04], Cesc Fabregas fouled the Catalan player.;"

Output:
[1877.04, 1878.04], Diego Costa anticipates.;
[1879.04, 1880.04], Jones can score.;
[1881.04, 1882.04], It ends up opening to Cuadrado.;
[1883.04, 1884.04], He asks for it back.;
[1884.04, 1885.04], Ivanovic.;
[1886.04, 1887.04], Ivanovic coming again.;
[1888.04, 1889.04], They asked for a hand.;
[1890.04, 1891.04], He does it clearly and ostensibly.;
[1892.04, 1893.04], The Serbian lateral.;
[1894.04, 1898.04], Who complains bitterly that this decision did not end with the 11 meters.;


'full_caption': "Branislav Ivanovic (Chelsea) takes a first-time shot from the edge of the box, but his effort is well blocked by the defender.",
    'echoes': "
    [1862.04, 1863.04], Chelsea had a peak of intensity.;
      [1864.04, 1865.04], The one where he scored the goal.; 
    [1867.04, 1869.04], He created a couple of very clear chances.; 
    [1870.04, 1876.04],  But he found it so easy again that the team has returned to another small valley in terms of intensity.; 
    [1877.04, 1878.04], Diego Costa anticipates.;
      [1879.04, 1880.04], Jones can score.; 
    [1881.04, 1882.04], It ends up opening square.; 
    [1883.04, 1884.04], He asks for it back.; 
    [1884.04, 1885.04], Ivanovic.; 
    [1886.04, 1887.04], Ivanovic coming again.; 
    [1888.04, 1889.04], They asked for a hand.;
      [1890.04, 1891.04], He does it clearly and ostensibly.; [1892.04, 1893.04], The Serbian side.; 
      [1894.04, 1898.04], Who complains bitterly that this decision did not end with the 11 meters.; 
      [1899.04, 1901.04], I honestly couldnt appreciate it live.; [1902.04, 1903.04],  Me neither.; 
      [1904.04, 1905.04], Let's watch some replay.; [1906.04, 1910.04], With that square ball back for Branislav Ivanovic's shot.; 
      [1911.04, 1912.04], He was jumping now, King.; [1912.04, 1913.04], He was jumping now, King.; 
      [1914.04, 1915.04], Jones was playing.; [1916.04, 1917.04], He tries to control her.; 
      [1918.04, 1921.04], Cesc Fabregas fouled the Catalan player.;",

    answer:
    [1877.04, 1878.04], Diego Costa anticipates.; #build-uo action
      [1879.04, 1880.04], Jones can score.;  #build-uo action
    [1881.04, 1882.04], It ends up opening square.; #build-uo action
    [1883.04, 1884.04], He asks for it back.;  #build-uo action
    [1884.04, 1885.04], Ivanovic.; #build-uo action
    [1886.04, 1887.04], Ivanovic coming again.; #action
    [1888.04, 1889.04], They asked for a hand.; #after action
      [1890.04, 1891.04], He does it clearly and ostensibly.; #after action supporting comments
        [1892.04, 1893.04], The Serbian side.; #after action supporting comments
      [1894.04, 1898.04], Who complains bitterly that this decision did not end with the 11 meters.; #after action supporting comments

"""


examples = [
    {
    'id':1,
    'caption': "Alexis Sanchez (Arsenal) collects a pass on the edge of the box and unleashes a shot which is brilliantly blocked by a defender.",
 'narration': "[1889.64, 1890.64], Delainey.; [1891.64, 1892.64], Delainey.; [1893.64, 1894.64], Delainey.; [1894.64, 1900.0800000000002], There is a run from Sanchez, but Sanchez shoots himself, Scott Dunn and Julian Spironi trip him up; [1900.0800000000002, 1908.2800000000002], Leaves the ball in the field. Cool episode from the Arsenal players. Great combination in attack; [1908.2800000000002, 1918.68], Crystal Palace's defensive players also rose to the occasion. Alexis Sanchez did not react; [1918.68, 1936.48], At his partner's run and hit with a turn himself. A long pass forward is received by Zaha. Panchan; [1937.48, 1951.0], He plays much deeper than he played before. I remember Panchan was almost a striker before that.;"
},
{
    'id': 2,
    'caption': "Branislav Ivanovic (Chelsea) takes a first-time shot from the edge of the box, but his effort is well blocked by the defender.",
    'narration': "[1862.04, 1863.04], Chelsea had a peak of intensity.; [1864.04, 1865.04], The one where he scored the goal.; [1867.04, 1869.04], He created a couple of very clear chances.; [1870.04, 1876.04], But he found it so easy again that the team has returned to another small valley in terms of intensity.; [1877.04, 1878.04], Diego Costa anticipates.; [1879.04, 1880.04], Jones can score.; [1881.04, 1882.04], It ends up opening square.; [1883.04, 1884.04], He asks for it back.; [1884.04, 1885.04], Ivanovic.; [1886.04, 1887.04], Ivanovic coming again.; [1888.04, 1889.04], They asked for a hand.; [1890.04, 1891.04], He does it clearly and ostensibly.; [1892.04, 1893.04], The Serbian side.; [1894.04, 1898.04], Who complains bitterly that this decision did not end with the 11 meters.; [1899.04, 1901.04], I honestly couldnt appreciate it live.; [1902.04, 1903.04],  Me neither.; [1904.04, 1905.04], Let's watch some replay.; [1906.04, 1910.04], With that square ball back for Branislav Ivanovic's shot.; [1911.04, 1912.04], He was jumping now, King.; [1912.04, 1913.04], He was jumping now, King.; [1914.04, 1915.04], Jones was playing.; [1916.04, 1917.04], He tries to control her.; [1918.04, 1921.04], Cesc Fabregas fouled the Catalan player.;",
},
{
'id': 3,
'caption': "Nemanja Matic (Chelsea) meets a cross from Eden Hazard, but he cannot keep his header on target. It goes high over the crossbar.",
'narration': "[1133.32, 1134.32], Fabregas.; [1135.32, 1138.32], There's a big space for him to put it in between Hart and his line.; [1139.32, 1141.32], And that penalty spot would be so dangerous.; [1142.32, 1145.32], 18 assists in the Premier League for Fabregas last season.; [1147.32, 1149.32], One directly here, Hazard.; [1150.32, 1151.32], Fabregas again.; [1152.32, 1156.32], Hazard, a great cross on the difficult one to manufacture; [1156.32, 1158.32], and towards Matic at the back.; [1158.32, 1161.32], Anything just a little bit too far in front of the near post.; [1162.32, 1164.32], Left it short on his first delivery.; [1164.32, 1166.32], Fabregas then plays a lovely little ball down the side.; [1167.32, 1170.32], So just too far ahead of the near post for him to be able to glance it.; [1174.32, 1176.32], That's been a high octane start.; [1177.32, 1179.32], True heart with not too much to do.; [1179.32, 1182.32], Begovic has made a brilliant beginning.; [1183.32, 1186.32], Jair is just waving to Hart to kick it somewhere else.; [1186.32, 1188.32], Jair is not too bothered, is he?; [1190.32, 1194.32], I think he wanted him to play the Fellaini role and he wasn't quite for having that.;"

},
]


# === FUNCTION: Generate Q&A from commentary ===
# def get_clean_commentary(full_caption, narration_caption):
#     prompt = f"""
# You are given the live narration of a soccer play, given by multiple sequential timestamps and a caption. You are also given a 
# full caption that summarizes the play. Your task is to maintain only the comments that are relevant to the action. Keep all comments 
# relevant to the action, including  the build up and after comments about the quality, progress, or insights about the play. 
# Only eliminate what is completely unrelated.


# Example:
# {example}

# For you asnwer omit the '#' comments of the example, and any other further comment besides the original captions that you consider relevant.


# Fullcaption:
# {full_caption}

# Narration:
# {narration_caption}
# """

def get_clean_commentary(full_caption, narration_caption):
    prompt = f"""
You are given:
- A 'full_caption': short text describing the main soccer action.
- A 'narration' list: timestamped commentary lines.

Task:
Return only the lines that are directly related to the action in 'full_caption'.

Keep:
- Lines describing the build-up to the action.
- Lines describing the action itself.
- Lines after the action that describe reactions, quality, or follow-up on the same play.

Remove:
- Lines about unrelated plays.
- Lines about different players or incidents not part of the action.
- Broadcast meta-comments like "Let's watch the replay" or "We'll see that again".
- Repeated filler lines that add nothing to the play.

Rules:
1. Keep the original timestamps and text exactly as in the input.
2. Keep the same order.
3. Do not add any new text or explanations.
4. Output only the filtered list, no extra comments.

Be sure to keep all lines to refer to the goal!

Now process:

Fullcaption:
{full_caption}

Narration:
{narration_caption}
"""
    

    prompt2 = f"""
    You are given:
    1. A `full_caption` describing a specific football/soccer action from a match.
    2. A `narration` list of lines in the format:
    [timestamp_start, timestamp_end], text;

    Your task:
    Select ONLY the lines from `narration` that directly describe:
    - The action in `full_caption`
    - The immediate build-up (passes, movement, positioning, assists, lead-up plays)
    - The immediate aftermath (result of the action, shot outcome, save, rebound, foul, VAR check, penalty call)

    Strict removal rules:
    - REMOVE lines about unrelated plays, different goals, or earlier/later events not connected to this specific action.
    - REMOVE lines that are only general praise/criticism or unrelated commentary.
    - REMOVE filler lines that don't advance the description of the play (e.g., crowd noise, weather, unrelated stats).
    - REMOVE repeated references to unrelated goals, even if the same player is mentioned.

    Timestamp continuity rule:
    - If a kept line has nearby lines within ±10 seconds that are part of the same continuous play, keep those too unless clearly irrelevant.
    - Always preserve chronological order.

    Player/team relevance:
    - Keep lines that mention the same player(s) or team as in `full_caption`, unless they are clearly about a different event.

    Output format (mandatory):
    - Output ONLY the kept lines, exactly as they appear in `narration`.
    - NO extra text, headings, explanations, quotes, or code blocks.
    - Each line must keep its original format: `[timestamp_start, timestamp_end], text;`
    - Separate lines with a newline.
    - Do not reorder timestamps.
    - If none of the timestamps are relevant output: "No relevant comments"

    ---

    Example 1:
    full_caption: "Diego Costa passes to Ivanovic, who takes a shot that is blocked by the goalkeeper."
    narration:
    [12.0, 14.0], Diego Costa moves forward with the ball.;
    [14.0, 16.0], He passes to Ivanovic on the right.;
    [16.0, 18.0], Ivanovic shoots towards goal!;
    [18.0, 20.0], The goalkeeper makes a fantastic save.;
    [22.0, 24.0], Replay of an earlier chance.;
    [25.0, 27.0], Crowd is chanting.

    Expected output:
    [12.0, 14.0], Diego Costa moves forward with the ball.;
    [14.0, 16.0], He passes to Ivanovic on the right.;
    [16.0, 18.0], Ivanovic shoots towards goal!;
    [18.0, 20.0], The goalkeeper makes a fantastic save.;

    ---

    Example 2:
    full_caption: "Messi dribbles past two defenders and scores."
    narration:
    [100.0, 102.0], Replay from last goal.;
    [102.0, 105.0], Messi gets the ball in midfield.;
    [105.0, 108.0], He dribbles past the first defender.;
    [108.0, 110.0], Past the second defender!;
    [110.0, 112.0], He shoots—goal!;
    [115.0, 118.0], Team celebrates wildly.;
    [125.0, 128.0], Substitution for the other team.

    Expected output:
    [102.0, 105.0], Messi gets the ball in midfield.;
    [105.0, 108.0], He dribbles past the first defender.;
    [108.0, 110.0], Past the second defender!;
    [110.0, 112.0], He shoots—goal!;
    [115.0, 118.0], Team celebrates wildly.;

    ---

    Now process the given input using the same rules and format.

    Fullcaption:
    {full_caption}
    Narration:
    {narration_caption}
    """

    
    response = ollama.chat(
        model='llama3:instruct',
        messages=[
            {"role": "user", "content": prompt2}
        ]
    )
    return response['message']['content'].strip()


if __name__ == "__main__": 
    output_path = 'results/validate_comments'
    input_path= 'results/all_plays.json'
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8' ) as file:
            data = json.load(file)

    results = []
    count =0
    for example in data:
    
        # if count%10 != 0:
        #     count+=1
        #     continue
        # count+=1
        if count > 100:
            break
        
        try:
            print(f"Processing commentary ID {example['id']}...")
            if(example['narration'] == ""):
                comm_output =""
            else:
                comm_output = get_clean_commentary(example['caption'], example['narration'])
            
            example['llama3_comment'] = comm_output

            results.append({
                "id": example["id"],
                "commentary": example["caption"],
                "commentary": comm_output
            })

            time.sleep(1)  # Respect model rate
        except Exception as e:
            print(f"Error processing ID {example['id']}: {e}")

    os.makedirs(output_path, exist_ok=True)

    # === OUTPUT: Save to JSON ===
    with open(f"{output_path}/validate_comments_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(f"{output_path}/all_plays_validate_comments.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(" generation complete. Results saved to results/validate_comments_output.json")

    df = pd.DataFrame(results)
    df.to_csv(f"{output_path}/validate_comment_output.csv", index=False)