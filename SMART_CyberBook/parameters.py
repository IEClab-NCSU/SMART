# ToDo: Move these parameters to a configuration file
'''
Parameter settings for SMART_CORE
'''
def get_smart_hyperparameters():
    strategyType = 'assessment'
    encodingType = 'tfidf'
    clusteringType = 'second'
    clusters = '10'
    save_clusterKeywordMapping = False
    save_assessmentSkillMapping = False
    save_paragraphSkillMapping = False
    inputFolder = None #since it is fetched directly from database
    outputFolder = 'output'
    n_run = ''

    return (strategyType, encodingType, clusteringType, clusters, save_clusterKeywordMapping, save_assessmentSkillMapping, save_paragraphSkillMapping, inputFolder, outputFolder, n_run)