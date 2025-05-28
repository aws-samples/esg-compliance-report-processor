import json
import logging
import re
import os 
import boto3

supplier_table = os.environ['SUPPLIER_TABLE']

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

ddb = boto3.resource('dynamodb')
bedrock_runtime = boto3.client('bedrock-runtime')

prompt= """You are a Social Sustainability Assistant working for AnyCompany Fashion.
Your task is to construct an email which will be sent to a supplier following evaluation of their audit
The email should have a professional tone and must match the format shown below in <format></format> tags:

The user will pass in all the information required to populate the email body.
Please Note.  The report Type will always be SMETA.

<format>
Good afternoon,

Thank you for sharing your latest ethichal audit and correction plan.

| Audit Date   | Type  | Auditing Firm        | Grading |
|--------------|-------|----------------------|---------|
|              | SMETA |                      |         |

Please see the below table for a breakdown of the grading and agreed timeframes to action remediation for each non-conformance.add()

| NC | DETAILS | GRADING | TIMEFRAME |
|----|---------|---------|-----------|
|    |         |         |           |

Evidence for remediation of each issue is expected to be shared within the given timeframes stipulated on the corrective action plan.
If improvements are not evidenced within the agreed timeframe, the factory grading will be downgraded to Caution / Emergency, in line with our Ethical Audit Policy.

Please provide an update in remedial progress by **15th April 2024.**

All information on our Ethical Audit Programme can be found on the _Supplier Website_ within the ESG section.

If you have any questions, please do not hesitate to ask.

Thank you in advance for your cooperation.
</format>

Here is an example email written by one of our existing Social Sustainability Assistants.
This is the base quality of email we expect you to produce.

"""

multi_issue_example = """Good afternoon Shaji,

Thank you for sharing your latest ethical audit and corrective action plan

Audit Date: 06/03/2024
Type: SMETA
Auditing firm: Intertek
Grading: Caution

The next audit is due by **6th March 2025.**

Please see the below table for a breakdown of the grading and agreed timeframes to action remediation for each non-conformance.

| NC | DETAILS | GRADING | TIMEFRAME |
|----|---------|---------|-----------|
| 1  | Systemic lapses / inadequate training for employees working in hazardous conditions e.g. chemicals / machinery etc.: No evidence of chemical training | Alert | 30 days |
| 2  | Isolated occurrence of inadequate fire exits. 1 fire door wedged open, must be closed to create compartmentation | Alert | 30 days |
| 3  | Failure to comply with legal requirements for building electrical safety e.g. inspections 4/7 outstanding C2 wiring issues to be repaired | Alert | 30 days |
| 4  | No / inadequate certificates for inspections of machinery, or machines not registered as required by law: racking inspection overdue Nov 2023 | Alert | 30 days |
| 5  | PPE provided but isolated incidents of workers not using PPE 1/1 worker not wearing a mask as is compulsory when operating pillow filling machine | Optimal | 30 days |
| 6  | Inadequate emergency exit signs. No emergency lights in Building 4 | Alert | 30 days |
| 7  | Inadequate process / systems for checking on legal right to work for all employees. 10/26 RTW docs were noted dated to confirm when checked - not validated | Caution | 60 days |


Evidence for remediation of each issue is expected to be shared within the given timeframes stipulated on the corrective action plan.
If improvements are not evidenced within the agreed timeframe, the factory grading will be downgraded to Caution / Emergency, in line with our Ethical Audit Policy.

Please provide an update in remedial progress by **15th April 2024.**

All information on our Ethical Audit Programme can be found on the _Supplier Website_ within the ESG section.

If you have any questions, please do not hesitate to ask.

Thank you in advance for your cooperation."""

zero_issue_example = """Good Afternoon Shaji,

Thank you for sharing your latest ethical audit and corrective action plan

Audit Date: 06/03/2024
Type: SMETA
Auditing firm: Intertek
Grading: Optimal

Zero non-conformances were indetified during the audit.

Thank you for your continued efforts in adhering to the Ethical Audit Policy.

All information on our Ethical Audit Programme can be found on the _Supplier Website_ within the ESG section.

If you have any questions, please do not hesitate to ask.

Thank you in advance for your cooperation."""

format = "Ensure the email is always wrapped in <format></format> tags."


def get_audit_issues(table_name: str, company_name: str, audit_date: str) -> dict:
    table = ddb.Table(table_name)
    
    response = table.query(
        KeyConditionExpression='#cn = :company AND begins_with(#sk, :date)',
        ExpressionAttributeNames={
            '#cn': 'Company Name',
            '#sk': 'AuditDateIssueNumber'
        },
        ExpressionAttributeValues={
            ':company': company_name,
            ':date': audit_date
        }
    )
    items = response['Items']
    return items

def filter_issues_response(response: dict) -> tuple[dict,dict]:
    issues = [item for item in response if "Issue Title" in item]
    supplier_info = [item for item in response if all(element in item for element in ["Country", "Site Contact"])]
    
    parsed_issues = [{key: value for key, value in issue.items() if key in ["Issue Title", "ESG Rating", "ESG Timescale"]} for issue in issues]
    
    return supplier_info[0], parsed_issues

def issues_to_markdown(issues: list[dict]) -> str:
    markdown_string = "| NC | DETAILS | GRADING | TIMEFRAME |"
    for i, issue in enumerate(issues, start=1):
        if issue["ESG Rating"] != 'N/A' and issue["ESG Timescale"] != 'N/A':
            title = issue["Issue Title"].strip()
            rating = issue["ESG Rating"].strip()
            timeframe = issue["ESG Timescale"].strip()
            record = f"| {i}  | {title} | {rating} | {timeframe} |"
            markdown_string += "\n" + record
        
    return markdown_string
        
def generate_email(supplier_info: dict, markdown_issues_table):
    ex1 = f"\n<example1>\n{multi_issue_example}\n</example2>"
    ex2 = f"\n<example2>\n{zero_issue_example}\n</example2>"
    system = prompt + ex1 + ex2
    
    if markdown_issues_table is None:
        markdown_issues_table = "There are no issues"

    system_list = [
        {
            "text": system
        },
        {
            "text": format
        }
    ]

    message_list = [
            {
                "role": "user",
                "content": [
                    {"text": "Can you generate an email body for me."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"text": "Please provide the supplier details"}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"text": json.dumps(supplier_info)}
                ]
            },
                   {
                "role": "assistant",
                "content": [
                    {"text": "Please provide the issue details and I will respond with the email body."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"text": markdown_issues_table}
                ]
            }
        ]
    inf_params = {"maxTokens": 2000, "topP": 0.4, "topK": 100, "temperature": 0.7}
    body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_list,
        "inferenceConfig": inf_params,
    }

    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId="us.amazon.nova-pro-v1:0"
    )
    print(response)

    streaming_body = response['body']
    response_content = streaming_body.read()
    response_text = response_content.decode('utf-8')

    response_json = json.loads(response_text)
    print(response_json)

    return  response_json['output']['message']['content'][0]['text']


def parse_email_response(response: str) -> str:
    try:
        start = re.search(r'<format>', response).span()[1]+1
        end = re.search(r'</format>', response).span()[0]-1
        return response[start:end]
    except AttributeError as e:
        logger.error(f"Could not parse email response: {e}")
        return response
    
def get_issues_markdown(table_name: str, company_name: str, audit_date: str) -> str:
    response = get_audit_issues(table_name, company_name, audit_date)
    _, issues = filter_issues_response(response)
    return issues_to_markdown(issues)

    
def handler(event, context):
    
    nc_uri = event["nc_uri"]
    section = event["section"]
    bucket = nc_uri.split('/')[2]
    import_uid = nc_uri.split('/')[3]
    company_name = event["company_name"]
    audit_date = event["audit_date"]
    clause = event["clause"]
    
    s3_client = boto3.client('s3')
    
    logger.info(f"Getting email for {company_name} on {audit_date}")
    response = get_audit_issues(supplier_table, company_name, audit_date)
    
    supplier_info, issues = filter_issues_response(response)
    logger.info(issues)
    markdown_issues_table = issues_to_markdown(issues)
    logger.info(f"{markdown_issues_table}")
    
    email_body = generate_email(supplier_info, markdown_issues_table)
    logger.info(f"{email_body}")
    
    email_markdown = parse_email_response(email_body)
    
    local_email_path = 'email.txt'
    remote_email_key = f"{import_uid}/email/email.txt"
    
      
    # Upload the content directly to S3
    s3_client.put_object(
        Bucket=bucket,
        Key=remote_email_key,
        Body=email_markdown.encode('utf-8')
    )
    
    return {
      "statusCode": 200,
      "nc_uri": nc_uri,
      "body": "Email markdown succesfully uploaded"
    }
    
