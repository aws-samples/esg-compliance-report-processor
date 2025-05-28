import json
import logging
import re
from typing import List, Dict

import boto3
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig
import yaml

from modules import partition_keys
from modules.tables import audit_table_factory

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def validate_table(table, structure: dict) -> bool:
    if all(value is None for value in structure.values()):
        raise ValueError("Table structure is empty, please alter config and include some way of identifying the table")
        
    if structure["key_terms"] is not None:
        table_text = table.get_text().lower()
        for string in structure["key_terms"]:
            if string.lower() not in table_text:
                return False
    return True


def validate_tables(tables: list, config: dict) -> dict:
    """
    Validate tables against config
    """
    titles = []
    valid_tables = {}
    new_config = {}
    # iterate through config tables and convert all keys to lower
    # add titles to a list for checking in next loop
    # init a new config dict for processing
    for table in config:
        obj = config[table]
        obj["index"] = None
        new_config[table.lower()] = obj
        if obj['structure']['title'] is None:
            continue
        titles.append(table.lower())
        
    del config
    # iterate document tables and make note of where titles are found
    for i in range(len(tables)):
        if tables[i].title is None:
            continue
        
        title = tables[i].title.text.lower()
        if title in titles:
            new_config[title]["index"] = i

    # iterate through config
    # validate each found table
    # if wrong table then check for nesting
    for table, obj in new_config.items():
        if new_config[table]["index"] is None:
            continue
        
        idx = obj["index"]
        structure = obj["structure"]
        
        if validate_table(tables[idx], structure):
            valid_tables[table] = tables[idx]
            logger.info(f"Table <{table}> is valid")
        else:
            logger.info(f"Table <{table}> not valid, checking for nesting")
            if validate_table(tables[idx+1], structure):
                valid_tables[table] = tables[idx+1]
                logger.info(f"Found valid table for <{table}> nested in parent table")
            elif validate_table(tables[idx-1], structure):
                valid_tables[table] = tables[idx-1]
                logger.info(f"Found valid table for <{table}> nested in parent table")

    missing_tables = {
        key: new_config[key]
        for key in set(new_config.keys()) - set(valid_tables.keys())
    }
    logger.info(f"Missing the following tables:\n{missing_tables}")
    
    for table in missing_tables:
        # we can assume the table either has no title or is not findable with the current method
        for _table in tables:
            if validate_table(_table, new_config[table]["structure"]):
                valid_tables[table] = _table
                logger.info(f"Table without title <{table}> is valid")
                logger.info(f"{_table.to_markdown()}")
                break
            
    missing_tables = {
        key: new_config[key]
        for key in set(new_config.keys()) - set(valid_tables.keys())
    }
            
    return valid_tables, missing_tables

def get_textract_only_tables(tables):
    with open("config/tables.yaml") as f:
        textract_tables = yaml.safe_load(f)
    
    json_tables = {}
    for table in tables:
        if table is None or table.title is None:
            continue
        title = table.title.text.lower()
        if title in textract_tables:
            json_tables[title] = table
    
    json_tables_combined = {}
    
    for title, table in json_tables.items():
        df = table.to_pandas()
        table_factory = audit_table_factory(title)
        json_tables_combined[title] = table_factory.build_table(table_data=df).to_json(orient="split")
    
    return json_tables_combined
    

def get_bedrock_tables(tables):
    with open('config/bedrock_tables.yaml') as f:
        bedrock_tables = yaml.safe_load(f)
    
    valid_tables, missing_tables = validate_tables(tables=tables, config=bedrock_tables)
            
    markdown_tables = {
        name: valid_tables[name].get_text(TextLinearizationConfig(table_linearization_format='markdown'))
        for name in valid_tables
    }
    return markdown_tables, missing_tables


def find_missing_data(pages, missing_tables) -> dict:
    found_pages = {}
    for table, obj in missing_tables.items():
        key_terms = obj["structure"]["key_terms"]
        for page in pages:
            text = page.get_text().lower()
            if any(term in text for term in key_terms):
                found_pages[table] = page.to_markdown()
                logger.info(f"Found data for table <{table}>")
    if non_findable := set(missing_tables.keys()) - set(found_pages.keys()):
        logger.info(f"Still missing data for the following tables, please reconsider their configuration:\n{non_findable}")
        
    return found_pages


def parse_response(response: str) -> dict:
    try:
        start = re.search(r"<response>", response).span()[1]+1
        end = re.search(r"</response>", response).span()[0]-1
        data = json.loads(response[start:end])
    except AttributeError as e:
        logger.info(f"### ERRONEOUS LLM RESPONSE:\n{response}")
        raise e
    #data = dict([ tuple(item.split(': ')) for item in response[start:end].split('\n') ])
    return data


def haiku_extract_from_table(table: str, queries: List[str], bedrock_runtime) -> Dict[str,str]:
    prompt = """You are a validation step in a data-science process, your responses should be consistent and reliable.
Your task is to analyze the markdown tables provided to you and extract any information the user asks for as a key value pair.
You should look through the table and consider its structure.  Some tables may have multi-hierarchical structures, others may be simple.
Your response should be in the following format enclosed in <response></response> tags:

<response>
{
    key1: value1,
    key2: value2,
    ...
}
</response>

Here is the table you must reference when responding to the information request:"""

    prompt += f"\n\n{table}"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ', '.join(queries)}
                ]
            }
        ],
        "temperature": 0.5,
        "max_tokens": 1000,
        "top_k": 100,
        "top_p": 0.1
    })
    
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        accept="application/json",
        contentType="application/json",
    )
    text = json.loads(response['body'].read())['content'][0]['text']
    # write function to catch a parse error and store the table somewhere
    return parse_response(text)

def sonnet_extract_from_page(page: str, queries: List[str], bedrock_runtime) -> Dict[str,str]:
    prompt = """You are a validation step in a data-science process, your responses should be consistent and reliable.
Your task is to analyze the markdown page provided to you and extract any information the user asks for as a key value pair.
You should look through the page and consider its structure.  The page may contain tables forms and seemingly unstructured data. You should make sense of all of this.
Your response should be in the following format enclosed in <response></response> tags:

<response>
{
    key1: value1,
    key2: value2,
    ...
}
</response>

Here is the page information you must reference when responding to the information request:"""

    prompt += f"\n\n{page}"

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "system": prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ', '.join(queries)}
                ]
            }
        ],
        "temperature": 0.5,
        "max_tokens": 1000,
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
    # write function to catch a parse error and store the table somewhere
    return parse_response(text)


def supplier_extract(supplier_uri: str) -> Dict:
    extractor = Textractor()
    bedrock_runtime = boto3.client("bedrock-runtime")

    document = extractor.start_document_analysis(
        file_source=supplier_uri,
        features=[TextractFeatures.TABLES],
        save_image=False
    )
    tables = document.tables
    pages = document.pages
    
    markdown_tables, missing_tables = get_bedrock_tables(tables=tables)
    page_data = find_missing_data(pages=pages, missing_tables=missing_tables)
    
    try:
        json_tables = get_textract_only_tables(tables=tables)
    except Exception as e:
        logger.info("Error getting textract only tables")
        logger.info(str(e))
        json_tables = {}
    
    logger.info("Opening bedrock tables config file")
    with open('config/bedrock_tables.yaml') as f:
        bedrock_tables = yaml.safe_load(f)

    logger.info("Extracting info from tables")
    table_query_dict = {
        name: haiku_extract_from_table(markdown_tables[name], bedrock_tables[name]["queries"], bedrock_runtime)
        for name in markdown_tables
    }
    logger.info("Extracting info from pages")
    page_query_dict = {
        name: sonnet_extract_from_page(page_data[name], bedrock_tables[name]["queries"], bedrock_runtime)
        for name in page_data
    }
    
    merged_dict = {}
    for page_info in page_query_dict.values():
        merged_dict.update(page_info)
    for table_info in table_query_dict.values():
        merged_dict.update(table_info)
        
    critical_keys = ["Company Name", "Date of Audit"]
    
    for key in critical_keys:
        if key not in merged_dict:
            raise ValueError("Critical key missing from bedrock extraction", key, supplier_uri)
    
    merged_dict["Date of Audit"] = partition_keys.format_audit_date(merged_dict["Date of Audit"])
    merged_dict["Company Name"] = partition_keys.format_company_name(merged_dict["Company Name"])
    
    string_dict = {
        re.sub(r'[^a-zA-Z0-9 ]', '', key).title(): str(value)
        for key, value in merged_dict.items()
    }
    string_dict["AuditDateIssueNumber"] = string_dict["Date Of Audit"]
    
    return {**string_dict, **json_tables}