import boto3
import json
import datetime
import sys
import os
import time
import PyPDF2
import csv
from io import BytesIO

from langchain import PromptTemplate, SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import AmazonKendraRetriever

s3 = boto3.client('s3')
s3_bucket = os.environ.get('s3_bucket') # bucket name
s3_prefix = os.environ.get('s3_prefix')
callLogTableName = os.environ.get('callLogTableName')
configTableName = os.environ.get('configTableName')
kendraIndex = os.environ.get('kendraIndex')
roleArn = os.environ.get('roleArn')
endpoint_name = os.environ.get('endpoint')

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({
            "inputs" : 
            [
                [
                    {
                        "role" : "system",
                        "content" : "You are a kind robot."
                    },
                    {
                        "role" : "user", 
                        "content" : prompt
                    }
                ]
            ],
            "parameters" : {**model_kwargs}})
        return input_str.encode('utf-8')
      
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]["content"]

content_handler = ContentHandler()
aws_region = boto3.Session().region_name
client = boto3.client("sagemaker-runtime")
parameters = {
    "max_new_tokens": 512, 
    "top_p": 0.9, 
    "temperature": 0.6
} 

llm = SagemakerEndpoint(
    endpoint_name = endpoint_name, 
    region_name = aws_region, 
    model_kwargs = parameters,
    endpoint_kwargs={"CustomAttributes": "accept_eula=true"},
    content_handler = content_handler
)

kendra = boto3.client("kendra", region_name=aws_region)
retriever = AmazonKendraRetriever(
    index_id=kendraIndex,
    region_name=aws_region,
    client=kendra
)
#relevant_documents = retriever.get_relevant_documents("what is the generative ai?")
#print('length of relevant_documents: ', len(relevant_documents))


def combined_text(title: str, excerpt: str) -> str:
    if not title or not excerpt:
        return ""
    return f"Document Title: {title} \nDocument Excerpt: \n{excerpt}\n"

def to_doc(doc) -> Document:    
    title = doc['DocumentTitle']['Text'] if doc['DocumentTitle']['Text'] else ""
    source = doc['DocumentURI']
    excerpt = doc['DocumentExcerpt']['Text']
    print('excerpt: ', excerpt)
    page_content = combined_text(title, excerpt)    
    metadata = {"source": source, "title": title}
    return Document(page_content=page_content, metadata=metadata)

def kendraQuery(query):
    response = kendra.query(QueryText=query, IndexId=kendraIndex)
    
    docs = []
    for query_result in response['ResultItems']:
        print('query_result: ', query_result)
        doc = to_doc(query_result)
        print('doc: ', doc)

        docs.append(doc)

    return docs
                

# store document into Kendra
def store_document(s3_file_name, requestId):
    documentInfo = {
        "S3Path": {
            "Bucket": s3_bucket,
            "Key": s3_prefix+'/'+s3_file_name
        },
        "Title": s3_file_name,
        "Id": requestId
    }

    documents = [
        documentInfo
    ]
        
    result = kendra.batch_put_document(
        Documents = documents,
        IndexId = kendraIndex,
        RoleArn = roleArn
    )
    print(result)

# load documents from s3
def load_document(file_type, s3_file_name):
    s3r = boto3.resource("s3")
    doc = s3r.Object(s3_bucket, s3_prefix+'/'+s3_file_name)
    
    if file_type == 'pdf':
        contents = doc.get()['Body'].read()
        reader = PyPDF2.PdfReader(BytesIO(contents))
        
        raw_text = []
        for page in reader.pages:
            raw_text.append(page.extract_text())
        contents = '\n'.join(raw_text)    
        
    elif file_type == 'txt':        
        contents = doc.get()['Body'].read()
    elif file_type == 'csv':        
        body = doc.get()['Body'].read()
        reader = csv.reader(body)        
        contents = CSVLoader(reader)
    
    print('contents: ', contents)
    new_contents = str(contents).replace("\n"," ") 
    print('length: ', len(new_contents))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    texts = text_splitter.split_text(new_contents) 
    print('texts[0]: ', texts[0])
        
    docs = [
        Document(
            page_content=t
        ) for t in texts[:3]
    ]
    return docs
              
def get_answer_using_template(query):    
    #relevant_documents = retriever.get_relevant_documents(query)
    relevant_documents = kendraQuery(query)
    print('length of relevant_documents: ', len(relevant_documents))
    print('relevant_documents: ', relevant_documents)    
    
    if(len(relevant_documents)==0):
        return llm(query)
    else:
        print(f'{len(relevant_documents)} documents are fetched which are relevant to the query.')
        print('----')
        for i, rel_doc in enumerate(relevant_documents):
            print(f'## Document {i+1}: {rel_doc.page_content}.......')
            print('---')

        prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Assistant:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        result = qa({"query": query})
        
        source_documents = result['source_documents']
        print(source_documents)

        return result['result']
        
def lambda_handler(event, context):
    print(event)
    userId  = event['user-id']
    print('userId: ', userId)
    requestId  = event['request-id']
    print('requestId: ', requestId)
    type  = event['type']
    print('type: ', type)
    body = event['body']
    print('body: ', body)

    global llm, kendra
    
    start = int(time.time())    

    msg = ""
    
    if type == 'text':
        text = body
        msg = get_answer_using_template(text)
        print('msg: ', msg)
            
    elif type == 'document':
        object = body
                    
        # store the object into kendra
        store_document(object, requestId)

        # summerization to show the document
        file_type = object[object.rfind('.')+1:len(object)]
        print('file_type: ', file_type)
            
        docs = load_document(file_type, object)
        prompt_template = """Write a concise summary of the following:

        {text}
                
        CONCISE SUMMARY """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        summary = chain.run(docs)
        print('summary: ', summary)

        msg = summary

    elapsed_time = int(time.time()) - start
    print("total run time(sec): ", elapsed_time)

    item = {
        'user-id': {'S':userId},
        'request-id': {'S':requestId},
        'type': {'S':type},
        'body': {'S':body},
        'msg': {'S':msg}
    }

    client = boto3.client('dynamodb')
    try:
        resp =  client.put_item(TableName=callLogTableName, Item=item)
    except: 
        raise Exception ("Not able to write into dynamodb")
        
    print('resp, ', resp)

    return {
        'statusCode': 200,
        'msg': msg,
    }