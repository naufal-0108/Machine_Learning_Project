import os, json, time
from tqdm import tqdm
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Literal
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from together import Together
from flask import Flask, render_template, request, jsonify, Response # Add Response

app = Flask(__name__, static_url_path='', static_folder='static')

db_password = os.environ.get("MONGODB_PASSWORD")

if db_password is None:
    raise ValueError("MONGODB_PASSWORD environment variable not set.")

uri = f"mongodb+srv://naufalfachri01:{db_password}@storage-1.shdojbu.mongodb.net/?retryWrites=true&w=majority&appName=storage-1"

client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.notes
assert db.list_collection_names() != [], "No such database or collection exists"
collections = db.documents

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
num_refining = 5
model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
temperature = 0.6
max_tokens = 1024

class RefinerSchema(BaseModel):
    refined_content: str
    
class EvaluatorSchema(BaseModel):
    status: Literal["PASS", "NEED REVISION"]
    revision: str

class DatabaseSchema(BaseModel):
    task: str
    title: Optional[str]
    content: Optional[str]
    
manager_prompt = """
You are Naufal’s personal assistant and the manager of two agents:
  • Refining Agent — polishes note content  
  • Database Agent — saves and retrieves notes  

ROLES
1. Personal Assistant
   – Engage proactively: suggest taking notes, design or discuss topics.
2. Agent Manager
   – Route tasks to the Refining or Database Agent using the prescribed JSON schemas.

WORKFLOW

1. NOTE CREATION  
   • If Naufal supplies content:
     1. Wrap it in a draft note (no JSON shown).  
     2. Ask “Refine this note?” (no JSON shown).  
        – Yes → output exactly `{refiner_agent_format}`  
        – No  → resume conversation.
   • If no content:
     1. Propose topics.  
     2. Ask Naufal to create a note now or later.  
        – Yes → show draft, allow edits; recommend "Refine?" as above (if Yes → Output exactly `{refiner_agent_format}`).
        – No  → continue chat; remind later.

2. NOTE REFINEMENT  
   • After receiving `{refiner_agent_format_resp}`, present refined text and ask satisfied or not.  
     – Yes → confirm and remember it.  
     – No  → output exactly `{refiner_agent_format}` again.

3. NOTE SAVING  
   • When Naufal agrees to save:
     – Output exactly `{database_agent_format_save}`  
     – After Database Agent confirms, notify Naufal.  
   • If he declines, remember content and continue.

4. NOTE RETRIEVAL  
   • On retrieval request → output exactly `{database_agent_format_retrieve}`  
   • When list returns → ask which title to act on.

5. NOTE DELETION  
   • Never suggest deletion. Only if requested:
     1. If no title list exists → output exactly `{database_agent_format_retrieve}`. Emit only the JSON object, no extra text.
     2. If list exists and title provided → output exactly `{database_agent_format_deletion}`.
"""

refiner_prompt = """
You are a competent refiner agent. You work with: your **manager** who is the main & frontline agent for Naufal and your other partner is **Database** agent who will save & retrive notes.
Your only main job/task is to refine the note's content that your manager orders you to do. You will be provided with 3 context: a task, a previous generation, and a revision. 
task is the order that your manager gives, previous generation is your previous refined content, and revision is the feedback from evaluator that you need to fix it. if the & previous generation & revision are none, that indicates it is the first time of the step.
You dont need to think very hard, just do it as best & fast as you can since your response will be evaluated by an evaluator AI later. You dont need to call the evaluator AI yourself, it will be called by the system.
Reflect to the evaluator's revision and improve your answer.

task:

Please refine this content:

{task}

previous generation:

{previous_generation}

revision:

{revision}

Only output JSON with the following schema:

{schema}
"""

evaluator_prompt = """
You are a competent evaluator. Your task is to assess the quality of a `previous_generation` provided in response to a specific `task` by a refiner agent who is naufal personal agent.
Please provide a detailed revision to improve the `previous_generation`. Your evaluation will focus on the following aspects:

**Evaluation Criteria:**
Carefully evaluate the `previous_generation` based on the following:
1.  **Structure & Clarity:** Is it well-organized, easy to understand, and logically structured?
2.  **Adding more nuance** Is it the content added some nuance to make it look more expressive?
2.  **Conciseness:** Is it to the point, without unnecessary verbosity?
3.  **Formatting:** Does the answer use formatting (e.g., bullet points, bolding, lists) effectively and appropriately?
4.  **Word Choice & Tone:** Is the language precise and suitable?

Only output the evaluation criteria with checker for every list above (complete or not) so that the refiner agent will know where to fix it for the revision.
if all the evaluation criteria are marked complete, set status to "PASS" else "NEED REVISION".

previous generation:

{previous_generation}

Decide if the content is "PASS" or "NEED REVISION", Only output JSON with the following schema:

{schema}
"""

database_prompt= """
You are a smart & competence database agent who works with: your **manager** who is the main & frontline agent for Naufal and your other partner is **Refiner** agent who will refine naufal's created note.
Your tasks will be saving & retrieving document from NoSQL (MongoDB) database. Your manager will give the order to you with this format: {manager_format}. 'Agent' means your name, 'task' means your task to save or retrive (task should be only either save or retrive not both),
'title' means title of the note (Optional), 'content' means content of the note (Optional). Pay close attention to the manager's order before answering!

Manager's order:

{task}

Only output in JSON format with following schema:

{db_schema} 
"""

def refiner_agent(task: Optional[str], previous_generation: Optional[str], revision: Optional[str], history: List, schema:Optional[Any]) -> str:
    """
    Refine a content based on the task, previous generation, revision
    """
    prompt = refiner_prompt.format(
        task=task,
        previous_generation=previous_generation,
        revision=revision,
        schema=schema.model_json_schema()
    )

    history += [{"role": "system", "content": prompt}]

    response = client.chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens, messages=history,
                                              response_format={"type": "json_object","schema": schema.model_json_schema(),})

    return response.choices[0].message.content, history

def evaluator_agent(previous_answer: Optional[str], schema: Optional[Any], history: List) -> str:

    

    history += [{"role": "system", "content": evaluator_prompt.format(previous_generation=previous_answer, schema=schema.model_json_schema())}]

    response = client.chat.completions.create(model=model, temperature=temperature, messages=history, max_tokens=max_tokens,
                                              response_format={"type": "json_object","schema": schema.model_json_schema(),})
    return response.choices[0].message.content, history


def database_agent(task: str, schema: Optional[Any]):

    message = [{"role": "system", "content": database_prompt.format(task=task, manager_format=database_agent_format_input, db_schema=database_agent_format_resp)}]

    response = client.chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens, messages=message,
                                              response_format={"type": "json_object","schema": schema.model_json_schema(),})
    
    return response.choices[0].message.content

def preprocessing(text):
    text = text.replace('"', "'")
    return text

refiner_agent_format = """{
    "agent": "Refiner Agent",
    "content": "content (keep formating)"
}
"""
refiner_agent_format_resp =  """{
    "sender": "Refiner Agent",
    "content": "content (keep formating)"
}"""

database_agent_format_retrieve = """{
    "agent": "Database Agent",
    "task": "retrieve",
    "title": "Title note" or "None"
}"""

database_agent_format_save = """{
    "agent": "Database Agent",
    "task": "save",
    "title": "Title note",
    "content": "Finalized Content (keep formating)"
}"""

database_agent_format_deletion = """{
    "agent": "Database Agent",
    "task": "delete",
    "title": "Title note"
}"""

database_agent_format_input = """
    "agent": "Database Agent",
    "task": ("save" or "retrieve" or "delete"),
    "title": "Title note" or "None"
    "content": "Content note" or "None"
"""
database_agent_format_resp = """
    "task": ("save" or "retrieve" or "delete"),
    "title": "Title note" or "None",
    "content": "Content note" or "None"
"""
message_history = [{"role": "system", "content": manager_prompt.format(refiner_agent_format=refiner_agent_format, refiner_agent_format_resp=refiner_agent_format_resp,
                                                                       database_agent_format_save=database_agent_format_save, database_agent_format_retrieve=database_agent_format_retrieve,
                                                                       database_agent_format_deletion=database_agent_format_deletion)},
                    {"role": "assistant", "content": "Hello Naufal 😊! I'm Atlas your AI assistant. How can I help you today?"}]

@app.route('/')
def index():
    """Renders the chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():

    global message_history, num_refining

    """Handles incoming chat messages and streams the response."""
    user_inputs = preprocessing(request.json.get('message'))

    print("=======================HUMAN MESSAGE============================")
    print(user_inputs)
    message_history.append({"role": "user", "content": user_inputs})

    if not user_inputs:
        return Response(json.dumps({"error": "No message provided"}), status=400, mimetype='application/json')
    
    print("========================AI MESSAGE==============================")

    manager_response = client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, messages=message_history).choices[0].message.content
    
    message_history.append({"role": "assistant", "content": manager_response})

    try:
        json_response = json.loads(manager_response, strict=False)
        agent = True

    except:
        agent = False


    if agent:
        
        def generate_tokens(json_response: dict):
             
            try:
                agent_name = json_response.get("agent").lower().strip()

                if agent_name == "refiner agent":

                    refine_agent_history = []
                    evaluator_agent_history = []

                    print("======================System Message============================")
                    print("Calling Refiner Agent...")
                    yield f"data: {json.dumps({'status': 'refiner'})}\n\n"
                    time.sleep(2.)
                    print("======================System Message============================")

                    yield f"data: {json.dumps({'status': 'refining'})}\n\n"
                    first_step, refine_agent_history = refiner_agent(task=json_response.get("content"), history=refine_agent_history, previous_generation=None, revision=None, schema=RefinerSchema)
                    prev_step = first_step

                    for num_ref in tqdm(range(5), desc="Refining content...", leave=True):
                        revision, evaluator_agent_history = evaluator_agent(
                            previous_answer=json.loads(prev_step)['refined_content'],
                            schema=EvaluatorSchema,
                            history=evaluator_agent_history
                        )

                        status = json.loads(revision)["status"]
                        feedback = json.loads(revision)["revision"]

                        print(status)

                        if status == "PASS":
                            break

                        prev_step, refine_agent_history = refiner_agent(
                            task=json_response.get("content"),
                            previous_generation=json.loads(prev_step)['refined_content'],
                            revision=feedback,
                            history=refine_agent_history,
                            schema=RefinerSchema
                        )

                        refine_agent_history.append({"role": "system", "content":json.loads(prev_step)['refined_content']})

                    refined_content = json.loads(prev_step)['refined_content']
                    refiner_agent_resp = str(dict({"sender": "Refiner_Agent", "content": refined_content}))

                    message_history.append({"role": "system", "content": refiner_agent_resp})


                    print("======================System Message============================")
                    print("Sending back to main agent...")
                    yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                    time.sleep(2.)
                    print("========================AI MESSAGE==============================")
                    yield f"data: {json.dumps({'status': 'thinking'})}\n\n"


                    llm_response_complete = ""

                    for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                        response = chunk.choices[0].delta.content or ""

                        llm_response_complete += response

                        yield f"data: {json.dumps({'token': response})}\n\n"
                
                    message_history.append({"role": "assistant", "content": llm_response_complete})

                    print()

                    yield f"data: {json.dumps({'end_of_stream': True})}\n\n"

                else:
                    print("======================System Message============================")
                    print("Calling Database Agent...")
                    yield f"data: {json.dumps({'status': 'database'})}"
                    print("======================System Message============================")

                    db_agent_resp = json.loads(database_agent(task=json_response, schema=DatabaseSchema), strict=False)
                    print(db_agent_resp, flush=True)

                    if db_agent_resp.get('task').lower().strip() == "save":

                        yield f"data: {json.dumps({'status': 'database-save'})}"

                        title_note = db_agent_resp.get('title').strip()
                        content_note = db_agent_resp.get('content').strip()

                        assert (title_note != "None") and (content_note != "None"), "Title and content must not be None"

                        new_collection = {"title": title_note, "content": content_note}
                        status = collections.insert_one(new_collection)

                        assert status.__getstate__()[-1]['_WriteResult__acknowledged'], "Fail to write to database"

                        feedback_db = {"role": "system", "content": str({"sender": "Database Agent", "saving_status": "The note has been successfully saved to the database!"})}
                        message_history.append(feedback_db)

                        time.sleep(1.)
                        print("======================System Message============================")
                        print("Sending back to main agent...")
                        yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                        time.sleep(1.)
                        print("========================AI MESSAGE==============================")
                        yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

                        llm_response_complete = ""
                        for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                            response = chunk.choices[0].delta.content or ""

                            llm_response_complete += response
                            print(response, end="", flush=True)
                            yield f"data: {json.dumps({'token': response})}\n\n"

                        message_history.append({"role": "assistant", "content": llm_response_complete})

                        print()

                        yield f"data: {json.dumps({'end_of_stream': True})}\n\n"

                    elif db_agent_resp.get('task').lower().strip() == "retrieve":

                        title_note = db_agent_resp.get('title').strip()

                        yield f"data: {json.dumps({'status': 'database-retrieve'})}"

                        if title_note.lower() == "none":

                            all_collections = [c.get("title") for c in collections.find({}, {"_id": 0, "title": 1})]

                            feedback_db = {"role": "system", "content": str({"sender": "Database Agent", "retreived_all_title_saved_notes": all_collections})}
                            message_history.append(feedback_db)

                            time.sleep(1.)
                            print("======================System Message============================")
                            print("Sending back to main agent...")
                            yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                            time.sleep(1.)
                            print("========================AI MESSAGE==============================")
                            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

                            llm_response_complete = ""
                            for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                                response = chunk.choices[0].delta.content or ""

                                llm_response_complete += response
                                print(response, end="", flush=True)
                                yield f"data: {json.dumps({'token': response})}\n\n"

                            message_history.append({"role": "assistant", "content": llm_response_complete})

                            print()

                            yield f"data: {json.dumps({'end_of_stream': True})}\n\n"
                        
                        else:
                            retrieved_docs = [c for c in collections.find({"title": title_note}, {"_id": 0})]

                            title_note = retrieved_docs[-1]["title"]
                            content_note = retrieved_docs[-1]["content"]

                            feedback_db =  {"role": "system", "content": str({"sender": "Database Agent", "retreived_titles": title_note, "retrieved_content": content_note})}
                            message_history.append(feedback_db)

                            time.sleep(1.)
                            print("======================System Message============================")
                            print("Sending back to main agent...")
                            yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                            time.sleep(1.)
                            print("========================AI MESSAGE==============================")
                            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
                            
                            llm_response_complete = ""
                            for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                                response = chunk.choices[0].delta.content or ""

                                llm_response_complete += response
                                print(response, end="", flush=True)
                                yield f"data: {json.dumps({'token': response})}\n\n"

                            message_history.append({"role": "assistant", "content": llm_response_complete})

                            print()

                            yield f"data: {json.dumps({'end_of_stream': True})}\n\n"

                    elif db_agent_resp.get('task').lower().strip() == "delete":

                        yield f"data: {json.dumps({'status': 'database-deletion'})}"

                        title_note = db_agent_resp.get('title').strip()
                        status_deletion = collections.find_one_and_delete({"title": title_note}, {"_id": 0})

                        if status_deletion:

                            feedback_db =  {"role": "system", "content": str({"sender": "Database Agent", "deleted_notes": status_deletion.get("title")})}
                            message_history.append(feedback_db)

                            time.sleep(1.)
                            print("======================System Message============================")
                            print("Sending back to main agent...")
                            yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                            time.sleep(1.)
                            print("========================AI MESSAGE==============================")
                            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
                            
                            llm_response_complete = ""
                            for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                                response = chunk.choices[0].delta.content or ""

                                llm_response_complete += response
                                print(response, end="", flush=True)
                                yield f"data: {json.dumps({'token': response})}\n\n"

                            message_history.append({"role": "assistant", "content": llm_response_complete})

                            print()

                        else:

                            feedback_db =  {"role": "system", "content": str({"sender": "Database Agent", "deleted_notes": "No note exists in database (check again)"})}
                            message_history.append(feedback_db)
                            print("======================System Message============================")
                            print("Sending back to main agent...")
                            yield f"data: {json.dumps({'status': 'manager'})}\n\n"
                            time.sleep(1.)
                            print("========================AI MESSAGE==============================")
                            yield f"data: {json.dumps({'status': 'thinking'})}\n\n"
                            
                            llm_response_complete = ""
                            for chunk in client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, stream=True, messages=message_history):

                                response = chunk.choices[0].delta.content or ""

                                llm_response_complete += response
                                print(response, end="", flush=True)
                                yield f"data: {json.dumps({'token': response})}\n\n"

                            message_history.append({"role": "assistant", "content": llm_response_complete})

                            print()

            except Exception as e:
                print(f"Error during streaming: {e}")
                # Yield an error message
                yield f"data: {json.dumps({'error': 'An error occurred during streaming.'})}\n\n"

        return Response(generate_tokens(json_response=json_response), mimetype='text/event-stream')

    else:
            
        def generate_tokens():

            try:

                print("========================AI MESSAGE==============================")


                llm_response_complete = ""
                for i, chunk in enumerate(manager_response.split(" ")):
                    if not chunk:
                        continue

                    chunk_to_yield = chunk

                    if i < len(manager_response) - 1:
                        chunk_to_yield += " "

                    llm_response_complete += chunk_to_yield

                    print(chunk_to_yield, end="", flush=True)
                    yield f"data: {json.dumps({'token': chunk_to_yield})}\n\n"
                    time.sleep(0.001)  # Simulate delay for streaming effect

                message_history.append({"role": "assistant", "content": llm_response_complete})

                yield f"data: {json.dumps({'end_of_stream': True})}\n\n"

            except Exception as e:
                print(f"Error during streaming: {e}")
                yield f"data: {json.dumps({'error': 'An error occurred during streaming.'})}\n\n"

        return Response(generate_tokens(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)