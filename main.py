import logging
from flask import Flask, jsonify, request, session
from datetime import timedelta
from flask_cors import CORS, cross_origin
from flask_session import Session
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

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
app.permanent_session_lifetime = timedelta(days=5)
app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=False  # Disable this for local development
)

Session(app)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "http://localhost:3000/"}})

client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client.login

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=3000, openai_api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key)
index_name = "recipes"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

system_prompt = (
        """Use the given context to generate custom recipes for the user. You should generate a unique recipe name, the ingredients needed to make
        the recipe, the steps to prepare the recipe, and the recipe's nutritional information along with the serving size, nothing else. Use the exact headings 'Recipe Name:', 'Ingredients:', 'Steps:', and 'Nutrition Info:' before each respective part of the response. 
        Ensure that all the ingredients include quantities and units. All the ingredients must be appropriate.
        Do not return any recipes that could be potentially dangerous or innapropriate.
        Under "Nutrition Info", include the following: "Serving Size", "Total Calories", "Carbohydrates", "Protein", and "Fat. Only use a colon ":" and nothing else to show the values for each of the above.
        They should be in the following format:
        Serving Size: X people
        Calories: X
        Carbohydrates: Xg
        Protein: Xg
        Fats: Xg
        You may only use hyphens "-" before each ingredient and you may not use hyphens "-" inside each ingredient."""
        """Dataset: \n{context}\n"""
    )

prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", system_prompt),
            ("human", "{input}")
        ],
    )

@app.route("/api/signup", methods=["POST"])
@cross_origin(supports_credentials=True)

#Allows users to create accounts, adds to db

def signup():
    session.permanent = True
    user = {
        "name": request.form.get("name"),
        "email": request.form.get("email"),
        "password": request.form.get("password"),
        "recipes": [],
    }

    user["password"] = pbkdf2_sha256.hash(user["password"])

    if db.users.find_one({"email": user['email']}):
         return jsonify({"error": "Account already exists"})

    db.users.insert_one(user)
    user_json = json.loads(json_util.dumps(user))
    del user['password']

    return jsonify(session["logged_in"]), 200

@app.route("/api/login", methods=["POST"])
@cross_origin(supports_credentials=True)

#Checks if account exists and if password matches 

def login():
    user = db.users.find_one({
        'email': request.form.get('email')
    })

    if user and pbkdf2_sha256.verify(request.form.get('password'), user['password']):
        user_json = json.loads(json_util.dumps(user))
        del user_json['password']
        session["logged_in"] = True
        session["user"] = user_json
        session.modified = True

        return jsonify(session["user"]), 200

    return jsonify({"error": "invalid login credentials"}), 401

@app.route("/api/logout", methods=["GET"])
@cross_origin(supports_credentials=True)

#Logs out user

def logout():
    session.pop("user", None)
    session["logged_in"] = False
    return jsonify(session["logged_in"]), 200

@app.route("/api/recipes", methods=["GET", "POST"])

#Fetches relevant documents from vector db for user's prompt, then generates recipe based on those documents

def recipes():
    if request.method == "POST":
        user_prompt = request.get_json()
        user_prompt = user_prompt.get("prompt")
            
        retriever = docsearch.as_retriever(search_time="similarity", search_kwargs={"k": 2})
        qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        chain = create_retrieval_chain(retriever, qa_chain)

        response = chain.invoke({"input": user_prompt})
        
        user = session.get('user', {})
        print(user)

        if user:
            email = user.get('email', 'email not found')
            print(email)
            curr_user = db.users.find_one({"email": email})

            if curr_user and "recipes" in curr_user:
                db.users.update_one(
                {"email": email},
                {"$push": {"recipes": response["answer"]}}
                )

        return jsonify(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
