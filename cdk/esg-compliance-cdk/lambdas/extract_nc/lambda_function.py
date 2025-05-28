import json
import os
import sys
import boto3
import pymupdf
import csv
import io
import re
from textractcaller.t_call import call_textract, Textract_Features
from trp.trp2 import TDocument, TDocumentSchema
from trp.t_pipeline import order_blocks_by_geo
import trp
from fuzzywuzzy import fuzz, process

bedrock_runtime = boto3.client('bedrock-runtime')
bedrock = boto3.client('bedrock')
s3_client = boto3.client("s3")
textract_client = boto3.client('textract')
ddb_client = boto3.client('dynamodb')

supplier_table = os.environ['SUPPLIER_TABLE']
compliance_grading_table = os.environ['GRADINGS_TABLE']

def get_rating(issue_title):
    
    response = ddb_client.scan(
        TableName=compliance_grading_table,
        FilterExpression='contains(#it, :value)',
        ExpressionAttributeNames={
            '#it':'Issue Title'
        },
        ExpressionAttributeValues={
            ':value': {'S':issue_title.strip()}
        }
    )

    items = response.get('Items',[])
    print(items)
    
    best_match = None
    highest_score = 0
    
    for item in items:
        db_issue_title = item.get('Issue Title',{}).get('S','')
        score = fuzz.partial_ratio(issue_title.lower(),db_issue_title.lower())
        print(score)
        if score > highest_score:
            highest_score = score
            best_match = item
    if best_match and highest_score > 80:
        return best_match
    else:
        return None
    

def order_document(document):
    

    # call textract
    textract_json = call_textract(input_document=document, features=[Textract_Features.FORMS, Textract_Features.TABLES], boto3_textract_client=textract_client)
    #load unordered document
    t_doc = TDocumentSchema().load(textract_json)
    # the ordered_doc has elements ordered by y-coordinate (top to bottom of page)
    ordered_doc = order_blocks_by_geo(t_doc)
    # send to trp for further processing logic
    trp_doc = trp.Document(TDocumentSchema().dump(ordered_doc))
        
   
    return trp_doc

def get_issues_timescale(ordered_doc):
    issues_timescale_list = []
    timescale_options = ["30 days", "60 days", "90 days", "120 days", "180 days", "365 days", "Immediate"]
    current_issue_title = None
    current_timescale = 'Other'
    table_data = []
    
    for page in ordered_doc.pages:
        
        try:
            for table in page.tables:
                for r, row in enumerate(table.rows):
                    for c, cell in enumerate(row.cells):
                        table_data.append("Table[{}][{}] = {}-{}".format(r,c, cell.text, cell.confidence))
                    
        except Exception as e:
            print(f"Error processing page: {e}")
            continue #Skip to next page if an error occurs 
        
        try:
            for field in page.form.fields:
                key = field.key.text if field.key else ""
                if key:
                    if field.key.text == "Issue Title":
                        if current_issue_title and current_timescale:
                            issues_timescale_list.append([current_issue_title, current_timescale])
                        current_issue_title = field.value.text
                        current_timescale = 'Other'
                    elif field.key.text in timescale_options and field.value.text == "SELECTED":
                        current_timescale = field.key.text
                    
        except Exception as e:
            print(f"Error processing page: {e}")
            continue #Skip to next page if an error occurs 

    if current_issue_title and current_timescale:
        issues_timescale_list.append([current_issue_title, current_timescale])
        
    return table_data, issues_timescale_list

def parse_issues(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Extract the part between <response> tags
    response_start = None
    response_end = None
    for i, line in enumerate(lines):
        if line.strip() == '<response>':
            response_start = i + 1
        elif line.strip() == '</response>':
            response_end = i
            break

    if response_start is None or response_end is None:
        return []

    # Remove leading and trailing double quotes from each line
    response_lines = []
    for line in lines[response_start:response_end]:
        cleaned_line = re.sub(r'^"(.+)"$', r'\1', line)
        response_lines.append(cleaned_line)

    response_content = '\n'.join(response_lines)

    # Split the content into individual issue entries
    issue_entries = re.findall(r'\[.*?\](?=,\n|\n])', response_content, re.DOTALL)

    issues = []
    for entry in issue_entries:
        # Remove the outer brackets
        entry = entry.strip()[1:-1]

        # Split the entry into its components
        components = re.split(r',\s*(?=[\'"])', entry)

        # Clean up each component by removing leading/trailing whitespace
        cleaned_components = [comp.strip().strip('\'"') for comp in components]

        # Remove any remaining brackets from the first and last components
        cleaned_components[0] = cleaned_components[0].lstrip('[').lstrip('"')
        cleaned_components[-1] = cleaned_components[-1].rstrip(']')

        issues.append(cleaned_components)

    return issues

def get_explanation(table_data,issues_timescale_list):
    
    table_input = ('\n\n').join(table_data)
    
    system_prompt = """
    You are a validation step in a data-science process,your responses should be consistent and reliable.
    
    You will be given a list of issue titles, and their remediation timescales.

    For each one you need to not whether it is an observation, good example, or non compliance, and then find the corresponding explanation. 

    Please do not paraphrase and ensure you get every single explanation. 
    Please do not add any additional information.
    Only focus on extracting the explanation and whether it is an observation, good example or non compliance. 

    Please provide a valid list of four-element arrays.
    Your response should be a valid array, enclosed in <format></format> tags:

    <response>
    [
        [<non-compliance | good-example | observation>, issue title, timescale, explanation],
        [<non-compliance | good-example | observation>, issue title, timescale, explanation],
        [<non-compliance | good-example | observation>, issue title, timescale, explanation],
        ...
    ]
    </response>
    """

    user_prompt = f"""
    Here is the list of issue_titles:
    f{issues_timescale_list}

    Here is the table data:
    {table_input}

    Please get the issue title and explanation for each issue.
    """
    
    body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "system": system_prompt,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ]
        }
        
    ],
     "temperature": 0.5,
    "max_tokens": 3000,
    "top_k": 100,
    "top_p": 0.1
    })

    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    text = json.loads(response['body'].read())['content'][0]['text']
    issue_timescale_explanation_list = parse_issues(text)
    
    return issue_timescale_explanation_list
    
def add_issue_to_dynamodb(issues_list,company_name,audit_date, clause,section):
    
    bedrock_validation_list = []
    count= 0
    count_issue = 0
    count_observation = 0
    count_exact = 0
    count_bedrock = 0 
    
    for item in issues_list: 
        count+=1
        
        if len(item) !=4:
            print(f'skipping malformed {item}')
            continue 

        nc_observation = item[0]
        issue_title = item[1]
        if '-' in issue_title:
            parts = issue_title.split('-', 1)
            if parts[0].strip().isdigit():
                issue_title = parts[1]
        timescale = item [2]
        explanation = item [3]
        audit_date_issue_no = audit_date+f'-{section}'+'#'+str(count)
        
        if 'non-compliance' in nc_observation.lower():
            count_issue+=1
             # updated_grading = best_match.get('Updated Grading'.{}).get('S')
            best_match = get_rating(issue_title)
            if best_match:
                count_exact+=1
                rating = best_match.get('Updated Grading',{}).get('S')
                esg_remediation_timescale = best_match.get('Resolution Window', {}).get('S',None)
                if rating:
                    if esg_remediation_timescale:
                        if esg_remediation_timescale.strip() == timescale.strip():
                            print('match!')
                            timescales_match = 'Yes'
                        else:
                            print('no match!')
                            timescales_match = 'No'
                        
                        
                        ddb_entry = {
                            'Company Name': {'S': company_name},
                            'AuditDateIssueNumber': {'S': audit_date_issue_no},
                            'Date Of Audit': {'S': audit_date},
                            'Clause': {'S':clause},
                            'Issue Type': {'S': 'non-compliance'},
                            'Issue Title': {'S': issue_title},
                            'Report Timescale': {'S': timescale},
                            'Report Explanation': {'S': explanation},
                            'ESG Rating': {'S': rating},
                            'ESG Timescale': {'S': esg_remediation_timescale},
                            'Exact Issue Title':{'S':'Yes'},
                            'Timescales Match': {'S': timescales_match}
                        }
                        
                        resp = ddb_client.put_item(
                            TableName= supplier_table,
                            Item=ddb_entry)
                            
                        
                        print(f"{ddb_entry} successfully written to {supplier_table}")        
                    else:
    
                        ddb_entry = {
                            'Company Name': {'S': company_name},
                            'AuditDateIssueNumber': {'S': audit_date_issue_no},
                            'Date Of Audit': {'S': audit_date},
                            'Clause': {'S':clause},
                            'Issue Type': {'S': 'non-compliance'},
                            'Issue Title': {'S': issue_title},
                            'Report Timescale': {'S': timescale},
                            'Report Explanation': {'S': explanation},
                            'ESG Rating': {'S': rating},
                            'ESG Timescale': {'S': 'N/A'},
                            'Exact Issue Title':{'S':'Yes'},
                            'Timescales Match': {'S': 'N/A'}
                        }
                        resp = ddb_client.put_item(
                            TableName= supplier_table,
                            Item=ddb_entry
                            )
                        print(f"{ddb_entry} successfully written to {supplier_table} with no ESG timescale")
                        
                else:
                    validation_entry = (audit_date_issue_no, nc_observation, issue_title, explanation, timescale)
                    print(f"No rating found for {issue_title} in wider table")
                    bedrock_validation_list.append(validation_entry)
                            
            else:
                validation_entry = (audit_date_issue_no, nc_observation, issue_title, explanation, timescale)
                print(f"Need to validate {issue_title} with wider table")
                bedrock_validation_list.append(validation_entry)
            
        elif 'observation' in nc_observation.lower():
            count_observation+=1
            #currently just storing observations - can expand to check within explanation if there is a 
            #perceived non-compliance within observation explanation 

            ddb_entry = {
                'Company Name': {'S': company_name},
                'AuditDateIssueNumber': {'S': audit_date_issue_no},
                'Date Of Audit': {'S': audit_date},
                'Clause': {'S':clause},
                'Issue Type': {'S': 'observation'},
                'Issue Title': {'S': issue_title},
                'Report Timescale': {'S': 'N/A'},
                'Report Explanation': {'S': explanation},
                'ESG Rating': {'S': 'N/A'},
                'ESG Timescale': {'S': 'N/A'},
                'Exact Issue Title':{'S':'No'},
                'Timescales Match': {'S': 'N/A'}
            }
            resp = ddb_client.put_item(
                TableName= supplier_table,
                Item=ddb_entry
                )
            print(f"Observation successfully written to {supplier_table}")
        
        elif 'good-example' in nc_observation.lower():
            count_observation+=1
            #currently just storing observations - can expand to check within explanation if there is a 
            #perceived non-compliance within observation explanation 

            ddb_entry = {
                'Company Name': {'S': company_name},
                'AuditDateIssueNumber': {'S': audit_date_issue_no},
                'Date Of Audit': {'S': audit_date},
                'Clause': {'S':clause},
                'Issue Type': {'S': 'good-example'},
                'Issue Title': {'S': issue_title},
                'Report Timescale': {'S': 'N/A'},
                'Report Explanation': {'S': explanation},
                'ESG Rating': {'S': 'N/A'},
                'ESG Timescale': {'S': 'N/A'},
                'Exact Issue Title':{'S':'No'},
                'Timescales Match': {'S': 'N/A'}
            }
            resp = ddb_client.put_item(
                TableName= supplier_table,
                Item=ddb_entry
                )
            print(f"Good Example successfully written to {supplier_table}")
        
    count_bedrock = len(bedrock_validation_list)   
    return bedrock_validation_list, count_issue, count_observation, count_exact, count_bedrock
                    

def handler(event,context):    
    # bucket = os.environ['REPORT_BUCKET']
    nc_uri= event["nc_uri"]
    clause = event["clause"]
    section = event["section"]
    company_name = event["company_name"]
    audit_date = event["audit_date"]
    
    ordered_doc = order_document(nc_uri)
    table_data, issues_timescale = get_issues_timescale(ordered_doc)
    
    if len(issues_timescale) > 0: 
        all_issues = get_explanation(table_data, issues_timescale)
        print(all_issues)
        bedrock_validation_list, count_issue, count_observation, count_exact, count_bedrock = add_issue_to_dynamodb(all_issues,company_name,audit_date,clause,section)
        
    else:
        all_issues = None
        bedrock_validation_list = None
        count_issue = 0
        count_observation = 0
        count_exact = 0
        count_bedrock = 0
    

    return {
        "nc_uri": nc_uri,
        "section": section,
        "clause": clause,
        "company_name": company_name,
        "audit_date": audit_date,
        "all_issues": all_issues,
        "unrated_issues": bedrock_validation_list,
        "count_issue": count_issue,
        "count_observation": count_observation,
        "count_exact": count_exact,
        "count_bedrock": count_bedrock
    }
    
