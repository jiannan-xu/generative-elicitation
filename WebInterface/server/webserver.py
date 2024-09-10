import os
import sys

# Ensure this can be run from the root directory.
sys.path.append('.')

import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pool_based_agent import PoolBasedAgent
from generative_questions_agent import GenerativeQuestionsAgent
from generative_edge_cases_agent import GenerativeEdgeCasesAgent
from generative_rag import GenerativeRag
import json
import random

load_dotenv()

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up Flask
app = Flask(__name__)

# specify the engine to use
ENGINE =  "gpt-4"

# save annotations in a directory
SAVE_DIR = f"annotations_{ENGINE}"
os.makedirs(SAVE_DIR, exist_ok=True)

# map query type to agent class
query_type_to_agent = {
    "Non-interactive": "non-interactive",
    "Supervised Learning": PoolBasedAgent,
    "Pool-based Active Learning": PoolBasedAgent,
    "Generative edge cases": GenerativeEdgeCasesAgent,
    "Generative open-ended questions": GenerativeQuestionsAgent,
    "Generative yes/no questions": GenerativeQuestionsAgent
}
# map query type to instructions
query_type_to_instruction = {
    "Non-interactive": "To the best of your ability, please explain all details about %task_description%, \
        such that someone reading your responses can understand and make judgments as close to your own as possible. \
        %noninteractive_task_description%\n<b>Note:</b> You will have up to 2 minutes to articulate your preferences. \
        Please try to submit your response within that time. After you submit, you will be taken to the final part of the study.",
    "Supervised Learning": "Try to answer in a way that accurately and comprehensively conveys your preferences, \
        such that someone reading your responses can understand and make judgments as close to your own as possible. \
        Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. \
        Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. \
        When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\
            \n<b>Note:</b> The chatbot will stop asking questions after 2 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Pool-based Active Learning": "Try to answer in a way that accurately and comprehensively conveys your preferences, \
        such that someone reading your responses can understand and make judgments as close to your own as possible. \
        Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. \
        Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. \
        When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\
        \n<b>Note:</b> The chatbot will stop asking questions after 2 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative edge cases": "This chatbot will ask you a series of questions about %task_description%. \
        Try to answer in a way that accurately and comprehensively conveys your preferences, \
        such that someone reading your responses can understand and make judgments as close to your own as possible. \
        Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. \
        Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. \
        When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\
        \n<b>Note:</b> The chatbot will stop asking questions after 2 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative open-ended questions": "This chatbot will ask you a series of questions about %task_description%. \
        Try to answer in a way that accurately and comprehensively conveys your preferences, such that someone reading your responses can understand and make judgments as close to your own as possible. \
        Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. \
        Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. \
        When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\
        \n<b>Note:</b> The chatbot will stop asking questions after 2 minutes, after which you can send your last response and you will be taken to the final part of the study.",
    "Generative yes/no questions": "This chatbot will ask you a series of questions about %task_description%. \
        Try to answer in a way that accurately and comprehensively conveys your preferences, \
        such that someone reading your responses can understand and make judgments as close to your own as possible. \
        Feel free to respond naturally (you can use commas, short phrases, etc), and press [enter] to send your response. \
        Note that the chatbot technology is imperfect, and you are free to avoid answering any questions that are overly broad or uncomfortable. \
        When interacting with the chatbot, please avoid asking follow-up questions or engaging in open-ended dialogue as the chatbot is unable to respond to you.\
        \n<b>Note:</b> The chatbot will stop asking questions after 2 minutes, after which you can send your last response and you will be taken to the final part of the study."
    }


def initialize_agent_by_query_type(query_type, problem_instance_filename, pool_fp, pool_al_sampling_type, pool_diversity_num_clusters):
    # Determine the question type based on the query type
    if query_type == "Generative yes/no questions":
        question_type = "yn"  # Yes/No type questions
    else:
        question_type = "open"  # Open-ended questions

    # Set the engine and cache file based on the query type
    if query_type == "Pool-based Active Learning":
        model_engine = "text-curie-001"
        # Cache responses for efficiency during pool-based active learning
        cache_file = f"{model_engine}-cache.jsonl"
    else:
        model_engine = ENGINE  # Use a predefined engine for other query types
        cache_file = None  # No cache required for non-pool-based tasks

    # Adjust temperature settings for generative edge cases
    if query_type == "Generative edge cases":
        temperature_setting = 0.8  # Higher temperature for more creative outputs
    else:
        temperature_setting = 0.0  # Lower temperature for more deterministic outputs

    # If the query type is associated with a predefined agent (a string), return it directly
    if isinstance(query_type_to_agent[query_type], str):
        return query_type_to_agent[query_type]

    # Special handling for Supervised Learning query type
    if query_type == "Supervised Learning":
        pool_al_sampling_type = "random"  # Force random sampling in supervised learning

    # Initialize and return the agent with the appropriate settings
    return query_type_to_agent[query_type](
        problem_instance_filename,
        model_engine,
        openai_cache_file=cache_file,
        question_type=question_type,
        pool_fp=pool_fp,
        pool_al_sampling_type=pool_al_sampling_type,
        pool_diversity_num_clusters=pool_diversity_num_clusters,
        temperature=temperature_setting,
    )

# load experiment type to prolific id mapping and prompt type to prompt mapping
experiment_type_to_prolific_id = json.load(open(f"{SAVE_DIR}/experiment_type_to_prolific_id.json"))
prompt_type_to_prompt = {}
for prompt_type in experiment_type_to_prolific_id:
    with open(f"human_exps_prompts/{prompt_type}.json") as f:
        prompt_type_to_prompt[prompt_type] = json.load(f)

def load_prolific_id_info_from_file():
    '''
    Load and organize data associated with participants identified by their Prolific IDs. 
    It uses this data to map each participant to their corresponding responses and experimental setup.
    '''
    # Stores user responses keyed by their Prolific ID.
    prolific_id_to_user_responses = {}
    # Maps each Prolific ID to their associated experimental setup, including the prompt type, query type, and initialized agent.
    prolific_id_to_experiment_type = {}

    # Load data for each participant from the annotations directory and organize it by Prolific ID.
    for filename in os.listdir(SAVE_DIR):
        if filename.endswith(".json") and filename != "experiment_type_to_prolific_id.json":
            prolific_id = os.path.split(filename)[-1].split(".json")[0]
            with open(os.path.join(SAVE_DIR, filename)) as f:
                prolific_id_to_user_responses[prolific_id] = json.load(f)

    # Map each Prolific ID to their associated experimental setup based on the experiment type and query type.
    for prompt_type in experiment_type_to_prolific_id:
        for query_type in experiment_type_to_prolific_id[prompt_type]:
            for prolific_id in experiment_type_to_prolific_id[prompt_type][query_type]:
                prolific_id_to_experiment_type[prolific_id] = {
                    "prompt": prompt_type_to_prompt[prompt_type],
                    "query_type": query_type,
                    "agent": initialize_agent_by_query_type(
                        query_type,
                        problem_instance_filename=os.path.join("gpt_prompts/", prompt_type, random.choice(os.listdir(f"gpt_prompts/{prompt_type}"))),
                        pool_fp=(prompt_type_to_prompt[prompt_type].get("full_data_path", None) if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_data_path", None)),
                        pool_al_sampling_type=("random" if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_al_sampling_type", None)),
                        pool_diversity_num_clusters=prompt_type_to_prompt[prompt_type].get("pool_diversity_num_clusters", None),
                    ),
                }
    # Return the organized data: user responses, experiment type to Prolific ID mapping, and Prolific ID to experiment type mapping.
    return prolific_id_to_user_responses, experiment_type_to_prolific_id, prolific_id_to_experiment_type

(
    prolific_id_to_user_responses,  # has fully completed at init
    experiment_type_to_prolific_id,  # has partially completed at init
    prolific_id_to_experiment_type,  # has partially completed at init
) = load_prolific_id_info_from_file()


# use html to display the experiment to the user
@app.route("/")
def home():
    '''
    The home function is associated with the root URL ("/") of the Flask application.
	When a user visits the root URL, the home function is triggered, and Flask renders the index.html template.
	The index.html file is typically used to define the homepage content of the web application.
    '''
    return render_template("index.html")


@app.route("/get_next_prompt", methods=["POST"])
def get_next_prompt():
    '''
    This decorator registers the /get_next_prompt URL with Flask, specifying that it handles POST requests.
    The function retrieves the Prolific ID from the request form and checks if the user already exists.
    '''
    prolific_id = request.form.get("prolific_id") # Retrieves the prolific_id from the form data submitted with the POST request.
    error = {}
    # Check if the user already exists
    if prolific_id in prolific_id_to_experiment_type:
        curr_prompt = prolific_id_to_experiment_type[prolific_id]["prompt"] # Retrieve the prompt associated with the user.
        curr_query_type = prolific_id_to_experiment_type[prolific_id]["query_type"] # Retrieve the query type associated with the user.
        prolific_id_to_user_responses[prolific_id] = {
            "prolific_id": prolific_id,
            "engine": ENGINE,
            "query_type": curr_query_type,
            "prompt": curr_prompt["prompt"],
            "conversation_history": [],
            "evaluation_results": [],
            "feedback": {},
        } # Initialize the user's responses with the prolific_id, engine, query type, prompt, conversation history, evaluation results, and feedback.
        error = {"error": "This username already exists"} # Set an error message if the user already exists.
    else:
        experiment_types_with_fewest_participants = [] # Initialize a list to store the experiment types with the fewest participants.
        min_num_participants = float("inf") # Initialize the minimum number of participants to infinity.
        prompt_types_to_consider = ["credit_card_preferences"] #["website_preferences"] # Define the prompt types to consider.
        for prompt_type in prompt_types_to_consider: # Iterate over the prompt types to consider.
            for query_type in experiment_type_to_prolific_id[prompt_type]: # Iterate over the query types for each prompt type.
                num_participants = len(experiment_type_to_prolific_id[prompt_type][query_type]) # Calculate the number of participants for the current experiment type.
                if num_participants < min_num_participants: # Check if the number of participants is less than the minimum number of participants.
                    experiment_types_with_fewest_participants = [(prompt_type, query_type)] # Update the experiment types with the fewest participants.
                    min_num_participants = num_participants # Update the minimum number of participants.
                elif num_participants == min_num_participants: # Check if the number of participants is equal to the minimum number of participants.
                    experiment_types_with_fewest_participants.append((prompt_type, query_type)) # Add the experiment type to the list of experiment types with the fewest participants.

        # sample experiment to run based on the experiment type with the fewest participants
        curr_prompt_type, curr_query_type = random.choice(experiment_types_with_fewest_participants) # Sample an experiment type to run based on the experiment type with the fewest participants.

        curr_prompt = prompt_type_to_prompt[curr_prompt_type] # Retrieve the prompt associated with the current prompt type.
        prolific_id_to_user_responses[prolific_id] = {
            "prolific_id": prolific_id,
            "engine": ENGINE,
            "query_type": curr_query_type,
            "prompt": curr_prompt["prompt"],
            "conversation_history": [],
            "evaluation_results": [],
            "feedback": {},
        } # Initialize the user's responses with the prolific_id, engine, query type, prompt, conversation history, evaluation results, and feedback.
        prolific_id_to_experiment_type[prolific_id] = {
            "prompt": curr_prompt,
            "query_type": curr_query_type,
            "agent": initialize_agent_by_query_type(
                curr_query_type,
                problem_instance_filename=os.path.join("gpt_prompts", curr_prompt_type, random.choice(os.listdir(f"gpt_prompts/{curr_prompt_type}"))),
                pool_fp=(prompt_type_to_prompt[prompt_type].get("full_data_path", None) if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_data_path", None)),
                pool_al_sampling_type=("random" if query_type == "Supervised Learning" else prompt_type_to_prompt[prompt_type].get("pool_al_sampling_type", None)),
                pool_diversity_num_clusters=curr_prompt.get("pool_diversity_num_clusters", None),
            ),
        } # Initialize the user's experiment type with the prompt and query type, and initialize the agent based on the query type.
        experiment_type_to_prolific_id[curr_prompt_type][curr_query_type].append(prolific_id) # Append the prolific_id to the experiment type with the fewest participants.

        json.dump(experiment_type_to_prolific_id, open(f"{SAVE_DIR}/experiment_type_to_prolific_id.json", "w"), indent=4) # Save the experiment type to prolific ID mapping to a JSON file.
    
    # Display the prompt to the user
    prompt_to_display = [
        curr_prompt["prompt"]["preamble"], # Display the preamble of the prompt.
        query_type_to_instruction[curr_query_type].replace("%task_description%", curr_prompt["prompt"]["task_description"]).replace("%noninteractive_task_description%", curr_prompt["prompt"].get("noninteractive_task_description", "")),
        curr_prompt["prompt"]["final"], # Display the final part of the prompt.
    ]
    # Convert the prompt to a string to display to the user.s
    prompt_to_display = "\n".join(prompt_to_display)
    agent = prolific_id_to_experiment_type[prolific_id]["agent"] # Retrieve the agent associated with the prolific_id.
    if type(agent) == str: # Check if the agent is a string.
        prolific_id_to_user_responses[prolific_id]["query_prompt"] = agent # Set the query prompt to the agent.
    else:
        prolific_id_to_user_responses[prolific_id]["query_prompt"] = agent.get_query_prompt() # Set the query prompt to the agent's query prompt.
    
    return jsonify({
        "prompt": prompt_to_display, # Display the prompt to the user.
        "evaluation_prompt": curr_prompt["prompt"]["evaluation"], # Display the evaluation prompt to the user.  
        "mode": "prompt" if curr_query_type == "Non-interactive" else "chat", # Display the mode to the user.
        **error,
    })

    # return jsonify({
    #     "prompt": prompt_to_display, # Display the prompt to the user.
    #     "evaluation_prompt": curr_prompt["prompt"]["evaluation"], # Display the evaluation prompt to the user.  
    #     "test_samples": curr_prompt["test_samples"], # Display the test samples to the user. (Update the test examples after preferences elicitation)
    #     "mode": "prompt" if curr_query_type == "Non-interactive" else "chat", # Display the mode to the user.
    #     **error,
    # })


@app.route("/update", methods=["POST"])
def update():
    """
    Sends user message (if exists) and queries active learning agent for next query
    """
    user_message = request.form.get("user_message") # Retrieve the user message from the form data submitted with the POST request.
    prolific_id = request.form.get("prolific_id") # Retrieve the prolific_id from the form data submitted with the POST request.
    if user_message: # Check if the user message exists.
        if prolific_id_to_experiment_type[prolific_id]["query_type"] != "Non-interactive": # Check if the query type is not non-interactive.
            previous_query = prolific_id_to_user_responses[prolific_id]["conversation_history"][-1]["message"] # Retrieve the previous query from the user responses.
            prolific_id_to_experiment_type[prolific_id]["agent"].add_turn(previous_query, user_message) # Add the user message to the agent's turns.
        assistant_display_timestamp = int(request.form.get("last_assistant_message_display_time")) # Retrieve the last assistant message display time from the form data submitted with the POST request.
        user_submission_timestamp = int(request.form.get("last_user_message_submission_time")) # Retrieve the last user message submission time from the form data submitted with the POST request.
        user_time_spent_on_message = user_submission_timestamp - assistant_display_timestamp # Calculate the time spent on the user message.
        prolific_id_to_user_responses[prolific_id]["conversation_history"].append({
            "sender": "user",
            "message": user_message,
            "time_spent_ms": user_time_spent_on_message,
            "display_time": assistant_display_timestamp,
            "submission_time": user_submission_timestamp,
        }) # Append the user message to the user responses, including the sender, message, time spent, display time, and submission time.
    query = None # Initialize the query to None.
    if not request.form.get("time_up"): # Check if the time is not up.
        query = prolific_id_to_experiment_type[prolific_id]["agent"].generate_active_query() # Generate the active query based on the prolific_id.
        prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "assistant", "message": query}) # Append the assistant message to the user responses.
    # # Store the generated test samples in the experiment type for this user
    # prolific_id_to_experiment_type[prolific_id]["prompt"]["test_samples"] = test_samples
    # curr_prompt["test_samples"] = test_samples
    return jsonify({"response": query}) # Display the query to the user.


@app.route("/update_user_response", methods=["POST"])
def update_user_response():
    user_message = request.form.get("user_message") # Retrieve the user message from the form data submitted with the POST request.
    prolific_id = request.form.get("prolific_id") # Retrieve the prolific_id from the form data submitted with the POST request.
    previous_query = prolific_id_to_user_responses[prolific_id]["conversation_history"][-1]["message"] # Retrieve the previous query from the user responses.
    prolific_id_to_experiment_type[prolific_id]["agent"].add_turn(previous_query, user_message) # Add the user message to the agent's turns.
    prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "user", "message": user_message}) # Append the user message to the user responses.
    return jsonify({"response": "done"}) # Display the response to the user.


@app.route("/get_next_query", methods=["POST"])
def get_next_query():
    prolific_id = request.form.get("prolific_id") # Retrieve the prolific_id from the form data submitted with the POST request.
    query = prolific_id_to_experiment_type[prolific_id]["agent"].generate_active_query() # Generate the active query based on the prolific_id.
    prolific_id_to_user_responses[prolific_id]["conversation_history"].append({"sender": "assistant", "message": query}) # Append the assistant message to the user responses.

    return jsonify({"response": query}) # Display the response to the user.


@app.route("/generate_test_examples", methods=["POST"])
def generate_test_examples():
    prolific_id = request.form.get("prolific_id")
    conversation_history = prolific_id_to_user_responses[prolific_id]["conversation_history"]
    print(conversation_history)
    reco_results = GenerativeRag(openai_api_key = openai.api_key, engine = ENGINE , conversation_history=conversation_history).generate_credit_card_recommendation()
    reco_rationale = reco_results["recommendation"]
    test_samples = [doc.page_content for doc in reco_results["source_documents"]]
    #test_samples = reco_results["source_documents"]
    print(test_samples)
    # assign the test samples to the experiment type for this user
    prolific_id_to_experiment_type[prolific_id]["prompt"]["test_samples"] = test_samples
    return jsonify({"response": "done", "reco_rationale": reco_rationale, "test_samples": test_samples})

@app.route("/save", methods=["POST"])
def save():
    prolific_id = request.form.get("prolific_id") # Retrieve the prolific_id from the form data submitted with the POST request.
    # make test_samples 
        
    with open(os.path.join(SAVE_DIR, f"{prolific_id}.json"), "w") as f: # Open the file associated with the prolific_id in write mode.
        json.dump(prolific_id_to_user_responses[prolific_id], f, indent=2) # Save the user responses to the file.
    return jsonify({"response": "done"}) # Display the response to the user.

# Evaluation and feedback submission routes
# @app.route("/submit_evaluation", methods=["POST"])
# def evaluation_submission():
#     prolific_id = request.form.get("prolific_id")
#     user_labels = []

#     # Assuming you have a function that generates credit card recommendations using RAG
#     test_samples = GenerativeRag(openai_api_key = openai.api_key, engine = ENGINE).recommend_credit_cards(prolific_id_to_user_responses[prolific_id]["conversation_history"])

#     # Store the generated test samples in the experiment type for this user
#     prolific_id_to_experiment_type[prolific_id]["prompt"]["test_samples"] = test_samples

#     # Iterate over the test samples and collect user feedback
#     for idx, test_sample in enumerate(test_samples):
#         user_labels.append({
#             "sample": test_sample,  # This is the credit card recommendation sample
#             "label": request.form.get(f"test-case-{idx}"),  # User's label (Yes/No)
#             "explanation": request.form.get(f"test-case-{idx}-explanation"),  # User's explanation
#         })

#     # Save the user's evaluation results
#     prolific_id_to_user_responses[prolific_id]["evaluation_results"] = user_labels
#     save()

#     return jsonify({"response": "done"})

@app.route("/submit_evaluation", methods=["POST"])
def evaluation_submission():
    generate_test_examples()
    prolific_id = request.form.get("prolific_id")
    user_labels = []
    # use test samples from the credit card recommendation results from RAG
    # pending editing
    for idx, test_sample in enumerate(prolific_id_to_experiment_type[prolific_id]["prompt"]["test_samples"]):
        user_labels.append({
            "sample": test_sample,
            "label": request.form.get(f"test-case-{idx}"),
            "explanation": request.form.get(f"test-case-{idx}-explanation"),
        })
    prolific_id_to_user_responses[prolific_id]["evaluation_results"] = user_labels
    save()
    return jsonify({"response": "done"})


@app.route("/submit_feedback", methods=["POST"])
def feedback_submission():
    prolific_id = request.form.get("prolific_id") # Retrieve the prolific_id from the form data submitted with the POST request.
    for feedback_type in request.form: # Iterate over the feedback types in the form data.
        if feedback_type.startswith("feedback_"): # Check if the feedback type starts with "feedback_".
            prolific_id_to_user_responses[prolific_id]["feedback"][feedback_type] = request.form.get(feedback_type)
    save() # Save the user responses.
    return jsonify({"response": "done"}) # Display the response to the user.


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
