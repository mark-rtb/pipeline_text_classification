"""
module with a function to clear the text
"""
import re


def text_preprocessing(text):
    """
    function to clear text from punctuation and non-standard characters
    on entry accepts:
        text ---------- str, cleaning text
    returns to output:
        clean_text ---- str, cleaned up text
    """
    
    reg = re.compile('[^\wA-Z ]')
    clean_text = reg.sub('', text)
    return clean_text


if __name__ == '__main__':
    pass
