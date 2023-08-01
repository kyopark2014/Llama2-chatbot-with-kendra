# Amazon JumpStart와 Amazon Kendra를 이용하여 Llama 2로 Chatbot 만들기

여기서는 Llama2의 LLM 모델을 이용하여 Question/Answering을 수행하는 Chatbot을 만듧니다. Question/Answering의 정확도를 높이기 위하여 관련된 문서를 업로드하면, Kendra로 분석하여 LLM이 Question/Answering 수행시 활용할 수 있습니다. 이와같이 Kendra를 이용하면, LLM의 token 사이즈를 넘어서는 긴문장을 활용하여 Question/Answering과 같은 Task를 수행할 수 있으며 환각(hallucination) 영향을 줄일 수 있습니다. 사용자가 파일을 로드하면 CloudFont와 API Gateway를 거쳐서 [Lambda (upload)](./lambda-upload/index.js)가 S3에 파일을 저장합니다. 저장이 완료되면 해당 Object의 bucket과 key 정보를 kendra에 전달합니다. 이후 사용자가 메시지를 질문을 하면, kendra가 가장 관련이 있는 문장을 LLM에 전달하고, 이를 이용하여 좀 더 정확한 답변을 얻을 수 있습니다. call log는 DynamoDB에 저장하여 확인할 수 있습니다.

전체적인 Architecture는 아래와 같습니다. 사용자가 파일을 업로드하면 Amazon S3에 저장된 후, kendra에 전달되어 symantic search에 활용되어집니다. 이후 사용자가 텍스트로 질문을 하면, CloudFront - API Gateway를 지나 [Lambda (chat)](./lambda-chat/lambda_function.py)에 텍스트가 전달됩니다. 이제 kendra를 통해 검색을 수행하여, 미리 입력한 문서중에서 가까운 문장을 얻습니다. 이후 Llama 2 LLM을 이용하여 답변을 얻습니다. 답변은 DynamoDB에 call log의 형태로 저장되어 추후 각종 통계정보나 결과를 분석하는데에 이용될 수 있습니다. Llama 2 LLM은 SageMaker Endpoint를 이용하여 LangChain 형식의 API를 통해 구현하였고, Chatbot을 제공하는 인프라는 AWS CDK를 통해 배포합니다. 

<img src="https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/e27d5831-91cc-44ce-ba6a-ff837b9f33c7" width="800">

문서파일을 업로드하여 Kendra에 저장하는 과정은 아래와 같습니다.

1) 사용자가 파일 업로드를 요청합니다. 이때 사용하는 Upload API는 [lambda (upload)](.lambda-upload/index.js)는 S3 presigned url을 생성하여 전달합니다.
2) 이후 presigned url로 문서를 업로드 하면 S3에 Object로 저장됩니다.
3) Chat API에서 request type을 'document'로 지정하면 [lambda (chat)](./lambda-chat/index.js)은 S3에서 object를 로드하여 텍스트를 추출합니다.
4) 추출한 텍스트를 Kendra로 전달합니다.

채팅 창에서 텍스트 입력(Prompt)를 통해 Kendra로 RAG를 활용하는 과정은 아래와 같습니다.
1) 사용자가 채팅창에서 질문(Question)을 입력합니다.
2) 이것은 Chat API를 이용하여 [lambda (chat)](./lambda-chat/index.js)에 전달됩니다.
3) lambda(chat)은 Kendra에 질문과 관련된 문장이 있는지 확인합니다.
4) Kendra로 부터 얻은 관련된 문장들로 prompt template를 생성하여 대용량 언어 모델(LLM) Endpoint로 질문을 전달합니다. 이후 답변을 받으면 사용자에게 결과를 전달합니다.
5) 결과는 DyanmoDB에 저장되어 이후 데이터 분석등의 목적을 위해 활용됩니다.


## 주요 구성

### IAM Role

IAM Role에서 아래와 같이 kendra에 대한 Permission을 추가해야 합니다.

```java
{
    "Effect": "Allow",
    "Action": [
        "kendra:*"
    ],
    "Resource": "arn:aws:kendra:[your-region]:[your-account-id]:index/[index-id]"
}]
```

이를 [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)에서는 아래와 구현할 수 있습니다.

```java
const region = process.env.CDK_DEFAULT_REGION;
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const kendraResourceArn = `arn:aws:kendra:${region}:${accountId}:index/${kendraIndex}`
if (debug) {
    new cdk.CfnOutput(this, `resource-arn-of-kendra-for-${projectName}`, {
        value: kendraResourceArn,
        description: 'The arn of resource',
    });
}
const kendraPolicy = new iam.PolicyStatement({
    resources: [kendraResourceArn],
    actions: ['kendra:*'],
});

roleLambda.attachInlinePolicy( // add kendra policy
    new iam.Policy(this, `kendra-policy-for-${projectName}`, {
        statements: [kendraPolicy],
    }),
);  
```

Kendra를 위한 trust policy는 아래와 같이 설정합니다.

```java
{
   "Version":"2012-10-17",
   "Statement":[
      {
         "Effect":"Allow",
         "Principal":{
            "Service":"kendra.amazonaws.com"
         },
         "Action":"sts:AssumeRole"
      }
   ]
}
```

따라서, [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)와 같이 "kendra.amazonaws.com"을 추가합니다.

```java
const roleLambda = new iam.Role(this, `role-lambda-chat-for-${projectName}`, {
    roleName: `role-lambda-chat-for-${projectName}`,
    assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("kendra.amazonaws.com")
    )
});
```

[Troubleshooting Amazon Kendra Identity and Access](https://docs.aws.amazon.com/kendra/latest/dg/security_iam_troubleshoot.html)와 같아 Kendra는 "iam:PassRole"을 포함하여야 합니다. 

```java
{
    "Action": [
        "iam:PassRole"
    ],
    "Resource": [
        "arn:aws:iam::[account-id]:role/role-lambda-chat-for-chatbot-with-kendra",
    ],
    "Effect": "Allow"
}
```

이를 [cdk-chatbot-with-kendra-stack.ts](./cdk-chatbot-with-kendra/lib/cdk-chatbot-with-kendra-stack.ts)에서는 아래와 같이 구현할 수 있습니다.

```java
const passRoleResourceArn = roleLambda.roleArn;
const passRolePolicy = new iam.PolicyStatement({
    resources: [passRoleResourceArn],
    actions: ['iam:PassRole'],
});
roleLambda.attachInlinePolicy(
    new iam.Policy(this, `pass-role-of-kendra-for-${projectName}`, {
        statements: [passRolePolicy],
    }),
);
```  

### LangChain으로 연결하기

LangChain을 이용해서 Llama 2에 연결하는 경우에 아래와 같이 endpoint_kwargs에 CustomAttributes를 추가합니다. 

```python
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
    "max_new_tokens": 256, 
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
```


### Kendra

### 문서 등록

S3에 저장된 문서를 kendra로 전달하기 위하여, 아래와 같이 문서에 대한 S3 정보를 kendra의 [batch_put_document()](https://docs.aws.amazon.com/kendra/latest/APIReference/API_BatchPutDocument.html)을 이용하여 전달합니다. 

```python
documentInfo = {
    "S3Path": {
        "Bucket": s3_bucket,
        "Key": s3_prefix+'/'+s3_file_name
    },
    "Title": "Document from client",
    "Id": requestId
}

documents = [
    documentInfo
]

kendra = boto3.client("kendra")
kendra.batch_put_document(
    Documents = documents,
    IndexId = kendraIndex,
    RoleArn = roleArn
)
```

업로드한 문서 파일에 대한 정보를 사용자에게 보여주기 위하여 아래와 같이 요약(Summerization)을 수행합니다.

```python
file_type = object[object.rfind('.') + 1: len(object)]
print('file_type: ', file_type)

docs = load_document(file_type, object)
prompt_template = """Write a concise summary of the following:

{ text }
                
CONCISE SUMMARY """

PROMPT = PromptTemplate(template = prompt_template, input_variables = ["text"])
chain = load_summarize_chain(llm, chain_type = "stuff", prompt = PROMPT)
summary = chain.run(docs)
print('summary: ', summary)

msg = summary
```


### Question/Answering

사용자가 채팅창에서 메시지를 입력할때 발생한 메시지는 아래처럼 query로 전달되고, [Kendra Retriever](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever)를 이용하여 get_relevant_documents()로 관련된 문장들을 kendra로부터 가져옵니다. 이때 가져온 문장이 없다면 Llama 2의 llm()을 이용하여 결과를 얻고, kendra에 관련된 데이터가 있다면 아래와 같이 template을 이용하여 [RetrievalQA](https://python.langchain.com/docs/modules/chains/popular/vector_db_qa)로 query에 대한 응답을 구하여 결과로 전달합니다.

```python
relevant_documents = retriever.get_relevant_documents(query)

if (len(relevant_documents) == 0):
    return llm(query)
else:
    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    { context }

    Question: { question }
    Assistant: """
    PROMPT = PromptTemplate(
        template = prompt_template, input_variables = ["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents = True,
        chain_type_kwargs = { "prompt": PROMPT }
    )
    result = qa({ "query": query })

    return result['result']
```

### AWS CDK로 인프라 구현하기

[CDK 구현 코드](./cdk-chatbot-llama2-kendra/README.md)에서는 Typescript로 인프라를 정의하는 방법에 대해 상세히 설명하고 있습니다.

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)



### CDK를 이용한 인프라 설치
[인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 


### 실행결과

채팅창에 "Tell me how to use ChatGPT in AWS."라고 입력합니다. 대규모 언어 모델은 가장 근사한 답을 찾아서 전달하므로 전형적인 환각(hallucination) 응답을 얻었습니다.

[그림]

[gen-ai-aws.pdf](./gen-ai-aws.pdf)을 다운로드 한 후에 채팅창 아래의 파일 아이콘을 선택하여 업로드합니다. 업로드가 끝나면 아래와 같이 "gen-ai-aws.pdf"을 요약하여 보여줍니다.

[그림]

"gen-ai-aws.pdf"는 AWS의 Generative AI에 대한 정보를 가지고 있으므로, 동일한 질문을 했을때에 아래와 같이 이전과 다른 정확한 답변을 얻을 수 있습니다.

[그림]

파일을 올리면 먼저 파일을 S3에 올리고, 이후로 kendra에 등록합니다. 업로드 한 파일의 내용을 확인하기 위하여 아래와 같이 요약(Summeraztion)을 수행합니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/74768245-6738-4a14-b942-cb6a9f39d252)

이후 아래와 같이 문서 내용에 대해 질문을 하면 답변을 얻을 수 있습니다.

![image](https://github.com/kyopark2014/question-answering-chatbot-with-kendra/assets/52392004/877d04a7-8190-43b5-a9fc-eab9e00ab990)

### 리소스 정리하기

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. [Cloud9 console](https://ap-northeast-2.console.aws.amazon.com/cloud9control/home?region=ap-northeast-2#/)에 접속하여 아래와 같이 삭제를 합니다.

```java
cdk destroy
```


## 결론

SageMaker JumpStart를 이용하여 대규모 언어 모델(LLM)인 LLama 2를 쉽게 배포하였고, Kendra를 이용하여 질문과 답변(Question/Answering)을 수행하는 chatbot의 기능을 향상시켰습니다. Amazon SageMaker JumpStart에서는 여러 종류의 대용량 언어 모델에서 한개를 선택하여 사용할 수 있습니다. 여기서는 Llama 2을 이용하여 RAG 동작을 구현하였고, 대용량 언어 모델의 환각(hallucination) 문제를 해결할 수 있었습니다. 또한 Chatbot 어플리케이션 개발을 위해 LangChain을 활용하였고, IaC(Infrastructure as Code)로 AWS CDK를 이용하였습니다. 대용량 언어 모델은 향후 다양한 어플리케이션에서 효과적으로 활용될것으로 기대됩니다. SageMaker JumpStart을 이용하여 대용량 언어 모델을 개발하면 기존 AWS 인프라와 손쉽게 연동하고 다양한 어플리케이션을 효과적으로 개발할 수 있습니다.





## Reference 

[Kendra - LangChain](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever)

[kendra_chat_anthropic.py](https://github.com/aws-samples/amazon-kendra-langchain-extensions/blob/main/kendra_retriever_samples/kendra_chat_anthropic.py)

[IAM access roles for Amazon Kendra](https://docs.aws.amazon.com/kendra/latest/dg/iam-roles.html)

[Adding documents with the BatchPutDocument API](https://docs.aws.amazon.com/kendra/latest/dg/in-adding-binary-doc.html)

[class CfnIndex (construct)](https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_kendra.CfnIndex.html)


### Chatbot 동작 시험시 주의할점

일반적인 chatbot들은 지속적인 세션을 유지 관리하기 위해서는 websocket 등을 사용하지만, 여기서 사용한 Chatbot은 API를 테스트하기 위하여 RESTful API를 사용하고 있습니다. 따라서, LLM에서 응답이 일정시간(30초)이상 지연되는 경우에 브라우저에서 답변을 볼 수 없습니다. 따라서 긴 응답시간이 필요한 경우에 CloudWatch에서 [lambda-chat](./lambda-chat/lambda_function.py)의 로그를 확인하거나, DynamoDB에 저장된 call log를 확인합니다.
