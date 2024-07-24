import re
from nltk.tokenize import sent_tokenize

def extract_key_sentences(context):

    keywords = [
    "measurement", "experiment", "theory", "values", "interaction", "proposal", 
    "verification", "quantum", "nonclassical", "weak values", "weak measurements", 
    "projector", "predictions", "dynamics", "eigenvalue", "imaginary part", 
    "real part", "inaccuracy", "entanglement", "counterfactual", 
    "observations", "data", "experimental", "realization", "implementation", 
    "analysis", "study", "research", "method", "technique", "procedure", 
    "findings", "results", "conclusion", "implications", "significance", 
    "accuracy", "precision", "instrumentation", "setup", "validation", 
    "hypothesis", "model", "simulation", "framework", "context", "overview", 
    "summary", "goal", "objective", "focus", "details", "elements", 
    "factors", "variables", "parameters", "conditions", "criteria", 
    "principles", "guidelines", "process", "system", "application", 
    "execution", "performance", "operation", "behavior", "function", 
    "task", "activity", "efficiency", "effectiveness", "skill", "expertise"
]

    sentences = sent_tokenize(context)
    
    key_sentences = [
        sentence for sentence in sentences
        if any(keyword in sentence for keyword in keywords)
    ]
    
    return key_sentences

def remove_references_and_emails(context):

    context = re.sub(r'\S+@\S+', '', context)

    context = re.sub(r'\[\d+\]', '', context)
    context = re.sub(r'\(\d+\)', '', context)

    context = re.sub(r'http\S+', '', context)
    
    return context

def format_references(context):
    references = re.sub(r'(\w+\. \w+\., "[^"]+", \w+\. \w+, vol\. \d+, pp\. \d+–\d+.)','', context)
    return references

def preprocess(context):
    context = remove_references_and_emails(context)
    references = format_references(context)
    return references


