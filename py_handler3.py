import boto3

def function_handler(event, context):
    logger.info('function_handler event: ' + repr(event))
