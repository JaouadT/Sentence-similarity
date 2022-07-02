import time
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import json
import requests
# import tensorflow as tf
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

app = Flask(__name__)
print("Loaded")


@app.route('/similarity_two_sentences', methods=['POST', 'GET'])
def get_similarity_two_sentences():
    jsonData = request.get_json()

    result = {}

    query_sentence = jsonData['sentence'][0]
    passage_sentences = jsonData['sentence'][1]

    query_embeddings = model.encode(query_sentence)
 
    passage_embedding = model.encode(passage_sentences)

    scores = util.cos_sim(query_embeddings, passage_embedding)

    # result['query_sentence'] = jsonData['sentence'][0]

    # di = {}
    # for i in range(len(jsonData['sentence'][1:])):
    #     tempD = {}
    #     tempD[jsonData['sentence'][1:][i]] = str(scores.numpy()[0][i])
    #     di[i] = tempD

    result['similarity'] = str(scores.numpy()[0][0])

    return Response(response=json.dumps({"response": result}), mimetype="application/json")

@app.route('/similarity_multiple_sentences', methods=['POST', 'GET'])
def get_similarity_multiple_sentences():
    jsonData = request.get_json()

    result = {}

    query_sentence = jsonData['sentence'][0]
    passage_sentences = jsonData['sentence'][1:]

    query_embeddings = model.encode(query_sentence)

    passage_embedding = model.encode(passage_sentences)

    scores = util.cos_sim(query_embeddings, passage_embedding)

    result['query_sentence'] = jsonData['sentence'][0]

    di = {}
    for i in range(len(jsonData['sentence'][1:])):
        tempD = {}
        tempD[jsonData['sentence'][1:][i]] = str(scores.numpy()[0][i])
        di[i] = tempD

    result['similarity'] = di

    return Response(response=json.dumps({"response": result}), mimetype="application/json")
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
