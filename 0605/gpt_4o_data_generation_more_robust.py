import openai
import json
import random
import time

# Set your OpenAI API key here
openai.api_key = "<MY API KEY>"

# Define the score descriptions based on the new rubric
score_descriptions = {
    1: """
SCORE OF 1: An essay in this category demonstrates very little or no mastery, and is severely flawed by
ONE OR MORE of the following weaknesses: develops no viable point of view on the issue, or provides
little or no evidence to support its position; the essay is disorganized or unfocused, resulting in a disjointed or
incoherent essay; the essay displays fundamental errors in vocabulary and/or demonstrates severe flaws in
sentence structure; the essay contains pervasive errors in grammar, usage, or mechanics that
persistently interfere with meaning.
""",
    2: """
SCORE OF 2: An essay in this category demonstrates little mastery, and is flawed by ONE OR MORE of
the following weaknesses: develops a point of view on the issue that is vague or seriously limited, and
demonstrates weak critical thinking, providing inappropriate or insufficient examples, reasons, or other
evidence to support its position; the essay is poorly organized and/or focused, or demonstrates serious
problems with coherence or progression of ideas; the essay displays very little facility in the use of
language, using very limited vocabulary or incorrect word choice and/or demonstrates frequent problems in
sentence structure; the essay contains errors in grammar, usage, and mechanics so serious that meaning is
somewhat obscured.
""",
    3: """
SCORE OF 3: An essay in this category demonstrates developing mastery, and is marked by ONE OR
MORE of the following weaknesses: develops a point of view on the issue, demonstrating some critical
thinking, but may do so inconsistently or use inadequate examples, reasons, or other evidence to support its
position; the essay is limited in its organization or focus, or may demonstrate some lapses in coherence or
progression of ideas displays; the essay may demonstrate facility in the use of language, but sometimes
uses weak vocabulary or inappropriate word choice and/or lacks variety or demonstrates problems in
sentence structure; the essay may contain an accumulation of errors in grammar, usage, and mechanics.
""",
    4: """
SCORE OF 4: An essay in this category demonstrates adequate mastery, although it will have lapses in
quality. A typical essay develops a point of view on the issue and demonstrates competent critical thinking,
using adequate examples, reasons, and other evidence to support its position; the essay is generally organized
and focused, demonstrating some coherence and progression of ideas exhibits adequate; the essay may
demonstrate inconsistent facility in the use of language, using generally appropriate vocabulary demonstrates
some variety in sentence structure; the essay may have some errors in grammar, usage, and mechanics.
""",
    5: """
SCORE OF 5: An essay in this category demonstrates reasonably consistent mastery, although it will
have occasional errors or lapses in quality. A typical essay effectively develops a point of view on the issue
and demonstrates strong critical thinking, generally using appropriate examples, reasons, and other evidence
to support its position; the essay is well organized and focused, demonstrating coherence and progression of
ideas; the essay exhibits facility in the use of language, using appropriate vocabulary demonstrates variety in
sentence structure; the essay is generally free of most errors in grammar, usage, and mechanics.
""",
    6: """
SCORE OF 6: An essay in this category demonstrates clear and consistent mastery, although it may have a
few minor errors. A typical essay effectively and insightfully develops a point of view on the issue and
demonstrates outstanding critical thinking, using clearly appropriate examples, reasons, and other evidence to
support its position; the essay is well organized and clearly focused, demonstrating clear coherence and
smooth progression of ideas; the essay exhibits skillful use of language, using a varied, accurate, and apt
vocabulary and demonstrates meaningful variety in sentence structure; the essay is free of most errors in
grammar, usage, and mechanics.
"""
}

# List of random topics for generating essays
topics = [
    "The impact of climate change on biodiversity",
    "The role of technology in modern education",
    "The benefits and drawbacks of remote work",
    "The importance of mental health awareness",
    "The future of renewable energy sources",
    "The effects of social media on society",
    "The challenges of space exploration",
    "The history and significance of human rights movements",
    "The influence of cultural diversity on creativity",
    "The ethical implications of artificial intelligence", 
    "Some of your friends perform community service. For example, some tutor elementary school children and others clean up litter. They think helping the community is very important. But other friends of yours think community service takes too much time away from what they need or want to do. \nYour principal is deciding whether to require all students to perform community service. \nWrite a letter to your principal in which you take a position on whether students should be required to perform community service. Support your position with examples.",
    "Some schools offer distance learning as an option for students to attend classes from home by way of online or video conferencing. Do you think students would benefit from being able to attend classes from home? Take a position on this issue. Support your response with reasons and examples.",
    "Some schools require students to complete summer projects to assure they continue learning during their break. Should these summer projects be teacher-designed or student-designed? Take a position on this question. Support your response with reasons and specific examples.",
    "Today the majority of humans own and operate cell phones on a daily basis. In essay form, explain if drivers should or should not be able to use cell phones in any capacity while operating a vehicle.",
    "When people ask for advice, they sometimes talk to more than one person. Explain why seeking multiple opinions can help someone make a better choice. Use specific details and examples in your response.",
    "Your principal has decided that all students must participate in at least one extracurricular activity. For example, students could participate in sports, work on the yearbook, or serve on the student council. Do you agree or disagree with this decision? Use specific details and examples to convince others to support your position. ",
    "Your principal is considering changing school policy so that students may not participate in sports or other activities unless they have at least a grade B average. Many students have a grade C average. \nShe would like to hear the students' views on this possible policy change. Write a letter to your principal arguing for or against requiring at least a grade B average to participate in sports or other activities. Be sure to support your arguments with specific reasons.",
    "Your principal is reconsidering the school's cell phone policy. She is considering two possible policies: \nPolicy 1: Allow students to bring phones to school and use them during lunch periods and other free times, as long as the phones are turned off during class time. \nPolicy 2: Do not allow students to have phones at school at all.\nWrite a letter to your principal convincing her which policy you believe is better. Support your position with specific reasons.",
    "Evaluate the pros and cons of studying Venus. Discuss the potential benefits and dangers of such a pursuit and provide reasons why studying Venus might be worthwhile",
    "Discuss the potential benefits and drawbacks of using technology to read and analyze human emotions in educational settings.",
    "Discuss the advantages and disadvantages of developing driverless cars. Present an argument for or against their widespread adoption.",
    "Discuss the pros and cons of the Electoral College versus the popular vote in presidential elections. Argue for which system you believe is better for democracy.",
    "Explain the advantages of limiting car usage. Discuss how reducing the number of cars on the road can benefit the environment, public health, and urban planning.",
    "Write an essay discussing the benefits of participating in volunteer programs that involve travel and adventure. Argue why such programs can be valuable for personal growth and global understanding.",
    "Discuss the importance of scientific evidence in debunking myths and misconceptions. Use the example of the 'Face on Mars' to argue why scientific explanations are crucial in understanding natural phenomena."
]

def generate_essays_on_topic(topic):
    essays = {}
    for score in range(1, 7):
        system_message = f"""
        You are an expert essay writer. Write an essay on the topic '{topic}' that matches the quality of a score {score} essay. Refer to the following guidelines for score {score}:
        {score_descriptions[score]}
        """

        success = False
        retries = 3
        while not success and retries > 0:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",  # or use gpt-4-turbo
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"Generate an essay on the topic '{topic}' that matches the quality of a score {score} essay."}
                    ],
                    max_tokens=1024
                )

                essays[score] = response['choices'][0]['message']['content']
                success = True
                time.sleep(1)  # To avoid hitting rate limits even if successful
            except Exception as e:
                print(f"Error generating essay for score {score} on topic '{topic}': {e}")
                retries -= 1
                time.sleep(1)  # To avoid hitting rate limits and provide a break before retrying
    return essays

def generate_essays(num_iterations):
    all_essays = []
    for i in range(num_iterations):
        topic = random.choice(topics)
        try:
            essays = generate_essays_on_topic(topic)
            all_essays.append({"topic": topic, "essays": essays})
            print(f"Generated essays for topic '{topic}' (iteration {i+1}/{num_iterations})")
        except Exception as e:
            print(f"Error generating essays for topic '{topic}' (iteration {i+1}/{num_iterations}): {e}")
        
        # Save intermediate results
        with open(f'generated_essays_intermediate_{i+1}.json', 'w') as f:
            json.dump(all_essays, f, indent=4)
        
        time.sleep(1)  # To avoid hitting rate limits

    return all_essays

# Generate essays 1500 times
num_iterations = 1 #1500
all_essays = generate_essays(num_iterations)

# Save generated essays to a JSON file
with open('generated_essays.json', 'w') as f:
    json.dump(all_essays, f, indent=4)
