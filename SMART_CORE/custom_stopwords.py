
"""
Additional stopwords that are abundant in the course text but are uninformative for SMART
"""	

def get_custom_stopwords():
    return {'correct', 'true', 'false', 'yes', 'following', 'mathrm'} # for oli_biology course