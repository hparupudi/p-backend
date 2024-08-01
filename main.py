from flask import Flask, jsonify, request, session
import os
from dotenv import load_dotenv
from passlib.hash import pbkdf2_sha256
import pymongo
import requests
import json
from bson import json_util, ObjectId

from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import create_retrieval_chain, SequentialChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.structured_output import create_structured_output_runnable
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")

load_dotenv()

client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client.login

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key or not pinecone_api_key:
    raise ValueError("Required API keys are missing")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=3000, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
index_name = "recipes"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

system_prompt = (
    """Use the given context to generate custom recipes for the user. You should generate a unique recipe name, the ingredients needed to make
    the recipe, the steps to prepare the recipe, and the recipe's nutritional information along with the serving size, nothing else. Use the exact headings 'Recipe Name:', 'Ingredients:', 'Steps:', and 'Nutrition Info:' before each respective part of the response. """
    """Dataset: \n{context}\n"""
)

prompt = ChatPromptTemplate.from_messages(
    messages=[
        ("system", system_prompt),
        ("human", "{input}")
    ],
)

@app.route("/api/recipes", methods=["GET", "POST"])
def recipes():
    if request.method == "POST":
        user_prompt = request.get_json()
        user_prompt = user_prompt.get("prompt")
        
        retriever = docsearch.as_retriever(search_time="similarity", search_kwargs={"k": 2})
        qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        chain = create_retrieval_chain(retriever, qa_chain)

        response = chain.invoke({"input": user_prompt})
        
        user_info = session.get('user', {})
        email = user_info.get('email', 'Email not found')

        curr_user = db.users.find_one({"email": email})

        if curr_user and "recipe" in curr_user:
            db.users.update_one(
                {"email": email},
                {"$push": {"recipe": response["answer"]}}
            )
        else:
            db.users.update_one(
                {"email": email},
                {"$set": {"recipe": [response["answer"]]}}
            )

        return jsonify(response["answer"])

@app.route("/api/signup", methods=["POST"])
def signup():
    user = {
        "name": request.form.get("name"),
        "email": request.form.get("email"),
        "password": request.form.get("password")
    }

    user["password"] = pbkdf2_sha256.hash(user["password"])

    if db.users.find_one({"email": user['email']}):
        return jsonify({"error": "Account already exists"})

    db.users.insert_one(user)
    user_json = json.loads(json_util.dumps(user))
    del user['password']
    session["logged_in"] = True
    session["user"] = user_json

    return jsonify(session), 200

@app.route("/api/login", methods=["POST"])
def login():
    user = db.users.find_one({
        'email': request.form.get('email')
    })

    if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
        user_json = json.loads(json_util.dumps(user))
        del user_json['password']
        session["logged_in"] = True
        session["user"] = user_json
        
        return jsonify(user_json), 200

    return jsonify({"error": "invalid login credentials"}), 401

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)