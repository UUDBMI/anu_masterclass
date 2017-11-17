#import everything that we will need
import pyConTextNLP
from pyConTextNLP import pyConTextGraph
from pyConTextNLP.itemData import itemData, contextItem, get_fileobj
from pyConTextNLP.display.html import mark_document_with_html
import os
import os.path
# useful utilities in RadNLP as well
import radnlp
import radnlp.view as rview
from radnlp.data import classrslts
import urllib
# our utilities from the class
import yaml
from nlp_pneumonia_utils import Annotation
from nlp_pneumonia_utils import AnnotatedDocument
from nlp_pneumonia_utils import read_brat_annotations
from nlp_pneumonia_utils import read_doc_annotations
from nlp_pneumonia_utils import read_annotations
from nlp_pneumonia_utils import calculate_prediction_metrics
from nlp_pneumonia_utils import mark_text
from nlp_pneumonia_utils import clearPyConTextRegularExpressions
from nlp_pneumonia_utils import pneumonia_annotation_html_markup
from nlp_pneumonia_utils import mark_document_with_html
from nlp_pneumonia_utils import view_single_sentence_graph
from nlp_pneumonia_utils import markup_sentence
from nlp_pneumonia_utils import markup_context_document
from nlp_pneumonia_utils import DocumentClassifier
# packages for interaction
from ipywidgets import interact, interactive, fixed
from IPython.display import display, HTML, Image
import ipywidgets


def get_items(_file):
    def get_fileobj(_file):
        if not urllib.parse.urlparse(_file).scheme:
            _file = "file://"+_file
        return urllib.request.urlopen(_file, data=None)
    f0 = get_fileobj(_file)
    
    context_items =    [contextItem((d["Lex"],
                                    d["Type"],
                                    r"%s"%d["Regex"],
                                    d["Direction"])) for d in yaml.load_all(f0)]
    return context_items

# This function let's us iterate through all documents and view the markup
def view_pycontext_graph(class_results, colors):
    @interact(i=ipywidgets.IntSlider(min=0, max=len(class_results)-1))
    def _view_markup(i):
        class_result = class_results[i]
        rview.markup_to_pydot(class_result)
        display(Image("tmp.png"))
        
        report_html = mark_document_with_html(class_result.context_document, colors = colors, default_color="black")
        
        display(HTML(report_html))
        
# This function let's us iterate through all documents and view the markup
def view_annotation_markup(anno_docs, colors):
    @interact(i=ipywidgets.IntSlider(min=0, max=len(anno_docs)-1))
    def _view_markup(i):
        report_html = pneumonia_annotation_html_markup(anno_docs[i])
        
        display(HTML(report_html))
        
def list_false_negatives(gold_docs, prediction_function):
    fn_docs={}
    for doc_name, gold_doc in gold_docs.items():
        gold_label=gold_doc.positive_label;
        pred_label = prediction_function(gold_doc.text)
        if gold_label==1 and pred_label==0:
            fn_docs[doc_name]=gold_doc            
    return fn_docs  

def list_false_positives(gold_docs, prediction_function):
    fn_docs={}
    for doc_name, gold_doc in gold_docs.items():
        gold_label=gold_doc.positive_label;
        pred_label = prediction_function(gold_doc.text)
        if gold_label==0 and pred_label==1:
            fn_docs[doc_name]=gold_doc            
    return fn_docs  

# prepare each of these for visualization
def marking_false_negatives(current_false_negatives, modifiers, targets):
    fn_report_results = []
    print('Marking up False Negatives')
    for anno_doc in current_false_negatives.values():
        report_context = markup_context_document(anno_doc.text, modifiers, targets)
        # package this up into a class that the RadNLP utilities can use
        results = classrslts(context_document=report_context, exam_type="Chest X-Ray", report_text=anno_doc.text, classification_result='N/A')
        fn_report_results.append(results)
    return fn_report_results
    
def marking_false_positives(current_false_positives, modifiers, targets):
    fp_report_results = []

    print('Marking up False Positives')
    for anno_doc in current_false_positives.values():
        report_context = markup_context_document(anno_doc.text, modifiers, targets)
        # package this up into a class that the RadNLP utilities can use
        results = classrslts(context_document=report_context, exam_type="Chest X-Ray", report_text=anno_doc.text, classification_result='N/A')
        fp_report_results.append(results)
    return fp_report_results