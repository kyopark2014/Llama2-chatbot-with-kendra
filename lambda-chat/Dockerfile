# lambda base image (python)
FROM amazon/aws-lambda-python:3.11

RUN /var/lang/bin/python3.11 -m pip install --upgrade pip
RUN /var/lang/bin/python3.11 -m pip install --upgrade boto3
RUN /var/lang/bin/python3.11 -m pip install PyPDF2
RUN /var/lang/bin/python3.11 -m pip install langchain

WORKDIR /var/task/lambda-chat

COPY lambda_function.py /var/task/
COPY . .

CMD ["lambda_function.lambda_handler"]